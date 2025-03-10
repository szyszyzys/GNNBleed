import os
from typing import Callable

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import GCN, GraphSAGE, GAT, GIN

DEFAULT_DATA_DIR = './'
DEFAULT_NUM_WORKERS = 4


def create_lightning_module(
        model_name: str,
        num_classes: int,
        dataset,
        h_dim: int = 256,
        pretrained: bool = False,
        ckpt: str = None,
        freeze_extractor: bool = False,
        num_layers=3,
        dropout=0.5,
        devices="cpu",
        *args,
        **kwargs,
):
    print(f'initializing model: {model_name}')
    if model_name.lower() == 'sage':
        _model = GraphSAGE(dataset.num_features, h_dim, dataset.num_classes, num_layers=num_layers, dropout=dropout)
    elif model_name.lower() == 'gcn':
        _model = GCN(dataset.num_features, h_dim, dataset.num_classes, num_layers=num_layers, dropout=dropout)
    elif model_name.lower() == 'gat':
        _model = GAT(dataset.num_features, h_dim, dataset.num_classes, num_layers=num_layers, dropout=dropout)
    elif model_name.lower() == 'gin':
        _model = GIN(dataset.num_features, h_dim, dataset.num_classes, num_layers=num_layers, dropout=dropout)

    if ckpt is not None:
        assert os.path.exists(ckpt), f"Failed to load checkpoint {ckpt}"
        checkpoint = torch.load(ckpt, map_location=torch.device(f'cuda:{devices[0]}'))
        pretrained_dict = checkpoint.state_dict()

        keys_to_remove = [k for k in pretrained_dict.keys() if "inspector" in k.lower()]
        for k in keys_to_remove:
            del pretrained_dict[k]

        pretrained_dict = {
            k.replace("_model.", ""): v
            for k, v in pretrained_dict.items()
            # if "fc" not in k and "classifier" not in k
        }
        _model.load_state_dict(pretrained_dict, strict=False)
    return LightningWrapper(_model, *args, **kwargs)


class TrainingPipeline:
    def __init__(
            self,
            model,
            datamodule,
            trainer: pl.Trainer,
    ):
        self.model = model
        self.datamodule = datamodule

        self.trainer = trainer

    def log_hparams(self):
        self.trainer.logger.log_hyperparams(self.model.hparams)

    def run(self):
        return self.trainer.fit(self.model, datamodule=self.datamodule)

    def test(self):
        return self.trainer.test(self.model, datamodule=self.datamodule)

    def predict(self, datamodule):
        return self.trainer.predict(self.model, datamodule=datamodule)


class StepTracker:
    def __init__(self):
        self.in_progress = False

    def start(self):
        self.cur_loss = 0
        self.in_progress = True

    def end(self, deduction: float = 0):
        self.in_progress = False


class LightningWrapper(pl.LightningModule):

    def __init__(
            self,
            model: torch.nn.Module,
            training_loss_metric: Callable = F.cross_entropy,
            optimizer: str = "Adam",
            lr_scheduler: str = "ReduceLROnPlateau",
            tune_on_val: float = 0.02,
            lr_factor: float = 0.5,
            lr_step: int = 10,
            batch_size: int = 64,
            lr: float = 0.03,
            momentum: float = 0.9,
            weight_decay: float = 5e-4,
            nesterov: bool = False,
            log_auc: bool = False,
            multi_class: bool = False,
            dataset_name=''
    ):
        super().__init__()
        # if we didn't copy here, then we would modify the default dict by accident
        self.save_hyperparameters(
            "optimizer",
            "lr_scheduler",
            "tune_on_val",
            "lr_step",
            "lr_factor",
            "lr",
            "momentum",
            "nesterov",
            "weight_decay",
            "batch_size",
        )

        self._model = model

        self._training_loss_metric = training_loss_metric
        self._val_loss_metric = training_loss_metric

        self._batch_transformations = []
        self._grad_transformations = []
        self._opt_transformations = []

        self._epoch_end_callbacks = []
        self._step_end_callbacks = []
        self._log_gradients = False
        self.dataset_name = dataset_name

        self.current_val_loss = 100
        self._on_train_epoch_start_callbacks = []
        self.automatic_optimization = False

        self.step_tracker = StepTracker()
        self.log_train_acc = True
        self.log_auc = log_auc

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.training_step_outputs = []

    def forward(self, data):
        res = self._model(data.x, data.edge_index)
        return res

    def predict_step(self, data, batch_idx=0, dataloader_idx=0):
        self._model.eval()
        with torch.no_grad():
            res = self(data)
        return F.softmax(res, dim=1)

    def predict_step_xy(self, x, edge_index, batch_idx=0, dataloader_idx=0):
        self._model.eval()
        with torch.no_grad():
            res = self._model(x, edge_index)
        return F.softmax(res, dim=1)

    def _compute_training_step(self,
                               data) -> dict:
        out = self(data)
        if self.dataset_name.startswith('twitch') or self.dataset_name.startswith(
                'flickr') or self.dataset_name.startswith('lastfm'):
            loss = self._training_loss_metric(out, data.y)
        else:
            loss = self._training_loss_metric(out[data.train_mask], data.y[data.train_mask])

        return {
            "loss": loss,
            "model_outputs": out,
            "target": data.y
        }

    def training_step(self, data) -> dict:
        if self.step_tracker.in_progress == False:
            self.step_tracker.start()
        training_step_results = self._compute_training_step(data)
        self.step_tracker.cur_loss += training_step_results["loss"].item()

        # get gradient
        self.manual_backward(training_step_results["loss"])

        # log gradient
        self.on_train_step_end()

        if self.log_train_acc:
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

        if self._log_gradients:
            grad_norm_dict = self.grad_norm(1)
            for k, v in grad_norm_dict.items():
                self.log(
                    f"gradients/{k}",
                    v,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )

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

    def configure_optimizers(self):
        if self.hparams["optimizer"] == "Adam":
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

        for transform in self._opt_transformations:
            transform(self)

        return self.optimizer

    def configure_lr_scheduler(self):
        self.lr_scheduler = None
        if self.hparams["lr_scheduler"] == "ReduceLROnPlateau":
            self.lr_scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.hparams["lr_factor"],
                patience=10,
                min_lr=1e-4,
                verbose=True,
            )

    def validation_step(self, data, batch_index):
        out = self.forward(data)
        if self.dataset_name.startswith('twitch'):
            loss = self._val_loss_metric(out, data.y)
            top1_acc = accuracy(out, data.y)[0]
            if self.log_auc:
                pred_list, true_list = auc_list(out, data.y)
            else:
                pred_list, true_list = None, None
        else:
            if data.val_mask is not None:
                loss = self._val_loss_metric(out[data.val_mask], data.y[data.val_mask])
            else:
                loss = self._val_loss_metric(out, data.y)

            top1_acc = accuracy(out[data.val_mask], data.y[data.val_mask])[0]
            if self.log_auc:
                pred_list, true_list = auc_list(out[data.val_mask], data.y[data.val_mask])
            else:
                pred_list, true_list = None, None
        self.validation_step_outputs.append({
            "batch/val_loss": loss,
            "batch/val_accuracy": top1_acc,
            "batch/val_pred_list": pred_list,
            "batch/val_true_list": true_list,
        }
        )
        return {
            "batch/val_loss": loss,
            "batch/val_accuracy": top1_acc,
            "batch/val_pred_list": pred_list,
            "batch/val_true_list": true_list,
        }

    def on_validation_epoch_end(self):
        # outputs is whatever returned in `validation_step`
        avg_loss = torch.stack([x["batch/val_loss"] for x in self.validation_step_outputs]).mean()
        avg_accuracy = torch.stack([x["batch/val_accuracy"]
                                    for x in self.validation_step_outputs]).mean()
        if self.log_auc:
            self.log_aucs(self.validation_step_outputs, stage="val")
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
        self.log("epoch/lr",
                 self.cur_lr,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        for callback in self._epoch_end_callbacks:
            callback(self)

    def test_step(self, data, batch_idx):
        out = self.forward(data)
        if self.dataset_name.startswith('twitch'):
            loss = self._val_loss_metric(out, data.y)
            pred_y = out.argmax(dim=1)
            top1_acc = accuracy(out, data.y)[0]
            # f1_score = torch.tensor(metrics.f1_score(data.y.cpu(), pred_y.cpu()))
            if self.log_auc:
                pred_list, true_list = auc_list(out.argmax(dim=1), data.y)
            else:
                pred_list, true_list = None, None
        else:
            loss = self._val_loss_metric(out[data.test_mask], data.y[data.test_mask])
            pred_y = out.argmax(dim=1)[data.test_mask]
            top1_acc = accuracy(out[data.test_mask], data.y[data.test_mask])[0]
            # f1_score = torch.tensor(metrics.f1_score(data.y.cpu(), pred_y.cpu()))
            if self.log_auc:
                pred_list, true_list = auc_list(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
            else:
                pred_list, true_list = None, None

        self.test_step_outputs.append({
            "batch/test_loss": loss,
            "batch/test_accuracy": top1_acc,
            # "batch/f1": f1_score,
            "batch/test_pred_list": pred_list,
            "batch/test_true_list": true_list,
        }
        )
        return {
            "batch/test_loss": loss,
            "batch/test_accuracy": top1_acc,
            "batch/test_pred_list": pred_list,
            "batch/test_true_list": true_list,
        }

    def on_test_epoch_end(self):
        avg_loss = torch.stack([x["batch/test_loss"] for x in self.test_step_outputs]).mean()
        avg_accuracy = torch.stack([x["batch/test_accuracy"]
                                    for x in self.test_step_outputs]).mean()
        # avg_f1 = torch.stack([x["batch/f1"]
        #                       for x in self.test_step_outputs]).mean()

        if self.log_auc:
            self.log_aucs(self.test_step_outputs, stage="test")
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

        # self.log("run/f1",
        #          avg_f1,
        #          on_epoch=True,
        #          prog_bar=True,
        #          logger=True)

    def log_aucs(self, outputs, stage="test"):
        pred_list = np.concatenate(
            [x[f"batch/{stage}_pred_list"] for x in outputs])
        true_list = np.concatenate(
            [x[f"batch/{stage}_true_list"] for x in outputs])

        aucs = []
        for c in range(len(pred_list[0])):
            fpr, tpr, thresholds = metrics.roc_curve(true_list[:, c],
                                                     pred_list[:, c],
                                                     pos_label=1)
            auc_val = metrics.auc(fpr, tpr)
            aucs.append(auc_val)

            self.log(
                f"epoch/{stage}_auc/class_{c}",
                auc_val,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        self.log(
            f"epoch/{stage}_auc/avg",
            np.mean(aucs),
            on_epoch=True,
            prog_bar=True,
            logger=True,
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


def auc_list(output, target):
    assert len(target.size()) == 2
    pred_list = torch.sigmoid(output).cpu().detach().numpy()
    true_list = target.cpu().detach().numpy()

    return pred_list, true_list
