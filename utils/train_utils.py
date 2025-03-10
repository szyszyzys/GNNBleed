import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from data_loader import GraphDataModule
from lighting_modules import create_lightning_module

'''
build model, dataload and trainer
'''
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def build_model(model_name='sage', dataset_name='twitch/ES', ckpt=None, lr=0.05, h_dim=256, num_layers=5, n_epoch=200,
                dropout=0.5, optimizer="Adam", remove_self_loop=False, log_root='./log',
                lr_scheduler="ReduceLROnPlateau",
                batch_size=32, gpuid=None, weight_decay=5e-4, momentum=0.9, patience=10, num_workers=4, arg=None):
    print("gpu id................")
    print(gpuid)
    if gpuid is None:
        gpuid = [0]
    args = {
        "model": model_name,
        "optimizer": optimizer,
        "scheduler": lr_scheduler,
        "gpuid": gpuid,
        "batch_size": batch_size,
        "tune_on_val": 0.02,
        "h_dim": h_dim,
        "ckpt": ckpt,
        "freeze_extractor": False,
        "log_auc": False,
        "n_epoch": n_epoch,
        "aug_hflip": True,
        "aug_crop": True,
        "aug_rotation": 0,
        "aug_colorjitter": None,
        "aug_affine": False,
        "n_accumulation_steps": 1,
        'num_layers': num_layers,
    }

    hparams = {
        "optimizer": optimizer,
        "lr": lr,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "nesterov": False,
        "lr_scheduler": lr_scheduler,
        "tune_on_val": 0.02,
        "batch_size": batch_size,
        "dataset_name": dataset_name
    }

    # build logger
    logger = TensorBoardLogger(
        "reconstruct_log", name=f'{args["model"]}/{args["optimizer"]}/{args["scheduler"]}')
    devices = [int(i) for i in gpuid]

    # if arg.defense_type == 'randedge' or arg.defense_type == 'lapgraph':
    #     implement_dp = True
    # else:
    implement_dp = False
    # build dataloader
    datamodule = GraphDataModule(
        dataset_name=dataset_name,
        batch_size=args["batch_size"],
        remove_self_loop=remove_self_loop,
        num_workers=num_workers,
        implement_dp=implement_dp,
        dp_epsilon=arg.dp_epsilon,
        noise_seed=arg.noise_seed,
        dp_delta=1e-5,
        noise_type='laplace',
        perturb_type=arg.defense_type
    )

    # loss function
    loss = torch.nn.CrossEntropyLoss()

    # build model
    model = create_lightning_module(
        model_name=model_name,
        dataset=datamodule,
        num_classes=datamodule.num_classes,
        h_dim=args["h_dim"],
        ckpt=args["ckpt"],
        freeze_extractor=args["freeze_extractor"],
        training_loss_metric=loss,
        num_layers=args['num_layers'],
        dropout=dropout,
        devices=devices,
        **hparams,
    )

    # lighting trainer
    trainer = pl.Trainer(
        check_val_every_n_epoch=10,
        deterministic=False,
        devices=devices,
        logger=logger,
        max_epochs=args["n_epoch"],
        accumulate_grad_batches=args["n_accumulation_steps"],
        callbacks=[EarlyStopping(monitor='epoch/val_loss', patience=patience)],
        default_root_dir=log_root
    )
    return model, datamodule, trainer
