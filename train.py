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

from data import LOBDataModule
from model import *
from utility import *
from eval import *

def train():
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


    with open("config/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = _cast_sci(cfg)

    model_cfg = cfg["model"]
    ckpt_cfg = cfg["checkpoint"]
    trainer_cfg = cfg["trainer"]

    set_random_seed(cfg["seed"])

    data_path = Path(cfg["data"]["path"])
    files = cfg["data"]["parquet_file"]
    if isinstance(files, str):
        data = pd.read_parquet(data_path / files)
    elif isinstance(files, list):
        data = [pd.read_parquet(data_path / f) for f in files]
    else:
        raise ValueError("cfg['data']['parquet_file'] 应该是 str 或 list[str]")

    dm = LOBDataModule(
        train_data=data[:-1],
        test_data=data[-1],
        dm_cfg=cfg["datamodule"],
        data_cfg=cfg["data"],
        task_type=model_cfg["task_type"],
    )
    dm.setup()

    monitor_metric = model_cfg["monitor_metric"]
    mode = model_cfg["mode"]

    model = DeepLOBLightning(
        input_width=model_cfg["input_width"],
        input_size=model_cfg["input_size"],
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        kernel_size=tuple(model_cfg["kernel_size"]),
        stride=tuple(model_cfg["stride"]),
        lr=model_cfg["lr"],
        neg_slope=model_cfg["neg_slope"],
        hidden_size=model_cfg["hidden_size"],
        lr_reduce_patience=model_cfg["lr_reduce_patience"],
        task_type=model_cfg["task_type"],
        monitor_metric=monitor_metric,
        mode=mode,
    )

    checkpoint_callback = ModelCheckpoint(
        mode=mode,
        monitor=monitor_metric,
        dirpath=ckpt_cfg["dirpath"],
        filename=ckpt_cfg["filename"],
        save_top_k=ckpt_cfg["save_top_k"],
    )

    early_stop_callback = EarlyStopping(
        mode=mode,
        monitor=monitor_metric,
        patience=cfg["early_stop"]["patience"],
        verbose=True,
    )

    logger = TensorBoardLogger(
        save_dir=cfg["logger"]["tensorboard"]["save_dir"],
        name=cfg["logger"]["tensorboard"]["name"],
    )

    trainer = Trainer(
        accelerator=trainer_cfg["accelerator"],
        devices=trainer_cfg["devices"],
        max_epochs=trainer_cfg["max_epochs"],
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=trainer_cfg["log_every_n_steps"],
    )

    trainer.fit(model, dm)

if __name__ == "__main__":
    train()
