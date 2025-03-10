import os
from typing import Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pytorch_lightning.cli import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy

from lighting_modules import StepTracker
from model import MLP


class MLP_Module(pl.LightningModule):
    def __init__(self, nfeat, nhid, nclass, n_layers=2, training_loss_metric: Callable = F.cross_entropy, ckpt=None):
        super().__init__()

        self.save_hyperparameters(
            "optimizer",
            "lr_scheduler",
            "tune_on_val",
            "lr_step",
            "lr_factor",
            "lr",
            "momentum",
            "weight_decay",
            "batch_size",
        )
        # # new PL attributes:
        self.train_acc = Accuracy('multiclass', num_classes=nclass)
        self.valid_acc = Accuracy('multiclass', num_classes=nclass)
        self.test_acc = Accuracy('multiclass', num_classes=nclass)

        # mlp model:
        self.model = MLP(nfeat, nhid, nclass, n_layers)

        self.step_tracker = StepTracker()
        self.automatic_optimization = False

        self._training_loss_metric = training_loss_metric
        self._val_loss_metric = training_loss_metric
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.training_step_outputs = []
        self._batch_transformations = []
        self._grad_transformations = []
        self._opt_transformations = []

        self._epoch_end_callbacks = []
        self._step_end_callbacks = []

        if ckpt is not None:
            assert os.path.exists(ckpt), f"Failed to load checkpoint {ckpt}"
            checkpoint = torch.load(ckpt, map_location=torch.device("cpu"))
            pretrained_dict = checkpoint.state_dict()
            pretrained_dict = {
                k.replace("_model.", ""): v
                for k, v in pretrained_dict.items()
                # if "fc" not in k and "classifier" not in k
            }
            pretrained_dict = {
                k.replace("module.", ""): v
                for k, v in pretrained_dict.items()
                # if "fc" not in k and "classifier" not in k
            }  # Unwrap data parallel model
            self.model.load_state_dict(pretrained_dict, strict=False)

    def forward(self, x):
        x = self.model(x)
        return x

    def _compute_training_step(self,
                               data) -> dict:
        x, y = data
        out = self(x)
        loss = self._training_loss_metric(out, y)

        return {
            "loss": loss,
            "model_outputs": out,
            "target": y
        }

    def training_step(self, batch, batch_idx):
        if self.step_tracker.in_progress == False:
            self.step_tracker.start()

        training_step_results = self._compute_training_step(batch)
        self.step_tracker.cur_loss += training_step_results["loss"].item()

        # get gradient
        self.manual_backward(training_step_results["loss"])

        # log gradient
        self.on_train_step_end()

        top1_acc = accuracy(
            training_step_results["model_outputs"],
            training_step_results["target"],
        )[0]
        self.log(
            "step/train_acc",
            top1_acc,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        return training_step_results

    def on_train_step_end(self) -> None:
        self.log("train_acc", self.train_acc.compute())
        self.train_acc.reset()

        self.optimizer.step()
        self.optimizer.zero_grad()

        for callback in self._step_end_callbacks:
            TERMINATE = callback(self, self.step_tracker)
            if TERMINATE:
                self.trainer.should_stop = True

        self.step_tracker.end()
        self.log(
            "step/train_loss",
            self.step_tracker.cur_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._val_loss_metric(logits, y)
        loss = nn.functional.cross_entropy(self(x), y)
        preds = torch.argmax(logits, dim=1)
        # self.valid_acc.update(preds, y)
        self.log("valid_loss", loss, prog_bar=True)

        top1_acc = accuracy(logits, y)[0]

        self.validation_step_outputs.append({
            "batch/val_loss": loss,
            "batch/val_accuracy": top1_acc
        })

        return {
            "batch/val_loss": loss,
            "batch/val_accuracy": top1_acc
        }

    def on_validation_epoch_end(self):
        # self.log("valid_acc", self.valid_acc.compute(), prog_bar=True)
        # self.valid_acc.reset()

        # outputs is whatever returned in `validation_step`
        avg_loss = torch.stack([x["batch/val_loss"] for x in self.validation_step_outputs]).mean()
        avg_accuracy = torch.stack([x["batch/val_accuracy"]
                                    for x in self.validation_step_outputs]).mean()
        self.validation_step_outputs.clear()

        self.current_val_loss = avg_loss

        if self.current_epoch > 0:
            if self.hparams["lr_scheduler"] == "ReduceLROnPlateau":
                self.lr_scheduler.step(self.current_val_loss)
            else:
                self.lr_scheduler.step()

        self.cur_lr = self.optimizer.param_groups[0]["lr"]

        self.log(
            "epoch/val_accuracy",
            avg_accuracy,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("epoch/val_loss",
                 avg_loss,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._val_loss_metric(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)

        top1_acc = accuracy(logits, y)[0]
        self.test_step_outputs.append({
            "batch/test_loss": loss,
            "batch/test_accuracy": top1_acc
        })
        return {
            "batch/test_loss": loss,
            "batch/test_accuracy": top1_acc
        }

    def on_test_epoch_end(self):
        avg_loss = torch.stack([x["batch/test_loss"] for x in self.test_step_outputs]).mean()
        avg_accuracy = torch.stack([x["batch/test_accuracy"]
                                    for x in self.test_step_outputs]).mean()
        self.test_step_outputs.clear()
        self.log("run/test_accuracy",
                 avg_accuracy,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log("run/test_loss",
                 avg_loss,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return avg_accuracy, avg_loss

    def configure_optimizers(self):
        if self.hparams["optimizer"] == "ADAM":
            self.optimizer = torch.optim.Adam(self.parameters(),
                                              lr=self.hparams["lr"])
        elif self.hparams["optimizer"] == "SGD":
            self.optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams["lr"],
                momentum=self.hparams["momentum"],
                weight_decay=self.hparams["weight_decay"],
                nesterov=self.hparams["nesterov"],
            )
        self.configure_lr_scheduler()

        return self.optimizer

    def configure_lr_scheduler(self):
        self.lr_scheduler = None
        if self.hparams["lr_scheduler"] == "ReduceLROnPlateau":
            self.lr_scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.hparams["lr_factor"],
                patience=2,
                min_lr=1e-4,
                verbose=True,
            )


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        if len(target.size()) == 1:  # single-class classification
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
        else:  # multi-class classification
            assert len(topk) == 1
            pred = torch.sigmoid(output).ge(0.5).float()
            correct = torch.count_nonzero(pred == target).float()
            correct *= 100.0 / (batch_size * target.size(1))
            res = [correct]
    return res


class SimiDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        xx, yy = self.x[idx], int(self.y[idx])
        return xx, yy


class SimilarityDataModule(pl.LightningDataModule):
    def __init__(self, x=None, y=None, x_test=None, y_test=None,
                 data_path='/content/drive/MyDrive/dl/link_steal_project/cora_attack_data/attack_1_x_train.pt'):
        super().__init__()
        if x == None:
            self.data_path = data_path
            self.x = torch.load(data_path).type(torch.float32)[:100]
            self.y = torch.tensor(torch.load(data_path[:-10] + 'y_train.pt'))[:100]
            self.x_test = torch.load(data_path[:-10] + 'x_test.pt').type(torch.float32)[:100]
            self.y_test = torch.tensor(torch.load(data_path[:-10] + 'y_test.pt'))[:100]

        else:
            self.x = x
            self.y = y
            self.x_test = x_test
            self.y_test = y_test
        self.trainset = SimiDataset(self.x, self.y)
        self.testset = SimiDataset(self.x_test, self.y_test)

    def setup(self, stage=None):
        self.train = self.trainset
        self.val = self.testset
        self.test = self.testset

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=64, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=64, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=64, num_workers=4)


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def get_auc(y_label, y_pred):
    # calculate scores
    ns_auc = roc_auc_score(y_label, y_pred)
    # summarize scores
    print('ROC AUC=%.3f' % (ns_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_label, y_pred)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='detection')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    return ns_auc
