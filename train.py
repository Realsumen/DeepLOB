# train.py
import pandas as pd, torch, pytorch_lightning as pl
from pathlib import Path
from typing import Tuple, List
import yaml
import re
from pathlib import Path
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from data import *
from model import *
from utility import *
from eval import *


def _cast_sci(obj):
    """
    递归把符合科学计数法格式的 str 转为 float
    """
    _sci_re = re.compile(
        r"""^[+-]?            # 可选正负号
            (?:\d+\.\d*|\d*\.\d+|\d+)  # 整数或小数
            [eE][+-]?\d+$    # e/E + 指数
        """,
        re.VERBOSE,
    )

    if isinstance(obj, dict):
        return {k: _cast_sci(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cast_sci(v) for v in obj]
    if isinstance(obj, str) and _sci_re.match(obj):
        return float(obj)
    return obj


def train():

    with open("config/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = _cast_sci(cfg)

    set_random_seed(cfg["seed"])

    data_path = Path(cfg["data"]["path"])
    files = cfg["data"]["parquet_file"]

    if isinstance(files, str):
        data = pd.read_parquet(data_path / files)
    elif isinstance(files, list):
        data = pd.concat(
            [pd.read_parquet(data_path / f) for f in files], ignore_index=True
        )
    else:
        raise ValueError("cfg['data']['parquet_file'] 应该是 str 或 list[str]")

    dm = LOBDataModule(
        data=data,
        batch=cfg["datamodule"]["batch_size"],
        val_ratio=cfg["datamodule"]["val_ratio"],
        random_split=cfg["datamodule"]["random_split"],
        data_cfg=cfg["data"],
    )
    dm.setup()

    model = DeepLOBLightning(
        input_width=cfg["model"]["input_width"],
        input_size=cfg["model"]["input_size"],
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        kernel_size=tuple(cfg["model"]["kernel_size"]),
        stride=tuple(cfg["model"]["stride"]),
        lr=cfg["model"]["lr"],
        neg_slope=cfg["model"]["neg_slope"],
        hidden_size=cfg["model"]["hidden_size"],
        lr_reduce_patience=cfg["model"]["lr_reduce_patience"]
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg["checkpoint"]["dirpath"],
        filename=cfg["checkpoint"]["filename"],
        monitor=cfg["checkpoint"]["monitor"],
        mode=cfg["checkpoint"]["mode"],
        save_top_k=cfg["checkpoint"]["save_top_k"],
    )

    early_stop_callback = EarlyStopping(
        monitor="val_acc",                          # 监控验证集准确率
        patience=cfg["early_stop"]["patience"],     # 连续 多 个 epoch 无提升就停
        mode="max",                                 # 准确率越大越好
        verbose=True,
    )

    logger = TensorBoardLogger(
        save_dir=cfg["logger"]["tensorboard"]["save_dir"],
        name=cfg["logger"]["tensorboard"]["name"],
    )

    trainer = Trainer(
        accelerator=cfg["trainer"]["accelerator"],
        devices=cfg["trainer"]["devices"],
        max_epochs=cfg["trainer"]["max_epochs"],
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=cfg["trainer"]["log_every_n_steps"],
    )

    trainer.fit(model, dm)

    evaluate_model_on_val(model, dm)


if __name__ == "__main__":
    train()
