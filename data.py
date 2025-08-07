import numpy as np, torch.nn as nn, pandas as pd, torch, pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from typing import List, Union
from torch.utils.data import Subset


class LOBDataset(Dataset):
    """
    支持传入单个 pd.DataFrame，或 List[pd.DataFrame]。
    每条样本由 (df_id, k) 唯一确定，其中 k 是各自 df 的行号。
    """

    def __init__(
        self,
        data: List[pd.DataFrame],
        task_type: str,
        data_cfg: dict,
    ):
        super().__init__()
        self.task_type = task_type
        self.seq_len = data_cfg["seq_len"]
        self.horizon = data_cfg["horizon"]
        self.alpha = data_cfg["alpha"]
        self.multiplier = data_cfg["multiplier"]
        self.features = data_cfg["feature_order"]
        self.use_rolling_mean = data_cfg["use_rolling_mean"]

        if isinstance(data, pd.DataFrame):
            data = [data]

        self.X_list, self.mid_list = [], []
        self.sample_map = []  # [(df_id, k), ...]

        for df_id, df in enumerate(data):
            X = df[self.features].ffill().values.astype(np.float32)
            mid = ((df["ask1"] + df["bid1"]) / 2).values.astype(np.float32)

            self.X_list.append(X)
            self.mid_list.append(mid)

            valid_idxs = np.arange(self.seq_len, len(df) - self.horizon)
            self.sample_map.extend([(df_id, int(k)) for k in valid_idxs])

    def __len__(self):
        return len(self.sample_map)

    def __getitem__(self, i):
        df_id, k = self.sample_map[i]
        X, mid_arr = self.X_list[df_id], self.mid_list[df_id]

        x_window = X[k - self.seq_len : k]
        if self.use_rolling_mean:
            past = mid_arr[k - self.seq_len : k].mean()
            future = mid_arr[k + 1 : k + 1 + self.horizon].mean()
        else:
            past = mid_arr[k]
            future = mid_arr[k + self.horizon]

        label = (future - past) / past * self.multiplier
        
        if self.task_type == "regression":
            return torch.from_numpy(x_window), torch.tensor(label, dtype=torch.float32)
        elif self.task_type == "classification":
            label = 2 if label > self.alpha else (0 if label < -self.alpha else 1)
            return torch.from_numpy(x_window), torch.tensor(label, dtype=torch.long)
        else:
            raise NotImplementedError(f"{self.task_type} is not support.")


class LOBDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data: Union[pd.DataFrame, List[pd.DataFrame]],
        test_data: Union[pd.DataFrame, List[pd.DataFrame]],
        dm_cfg: dict,
        data_cfg: dict,
        task_type: str
    ):
        super().__init__()

        self.batch = dm_cfg["batch_size"]
        self.val_ratio = dm_cfg["val_ratio"]
        self.random_split = dm_cfg.get("random_split", False)
        self.num_workers = dm_cfg.get("num_workers", 4)
        self.task_type = task_type

        self.train_data, self.test_data = train_data, test_data
        self.data_cfg = data_cfg

    def setup(self, stage=None):
        if stage in (None, "fit"):
            full_ds = LOBDataset(self.train_data, self.task_type, self.data_cfg)
            if self.random_split:
                n_val = int(len(full_ds) * self.val_ratio)
                self.train_set, self.val_set = torch.utils.data.random_split(
                    full_ds, [len(full_ds) - n_val, n_val]
                )
            else:
                n = len(full_ds)
                n_val = int(n * self.val_ratio)
                self.train_set = Subset(full_ds, list(range(0, n - n_val)))
                self.val_set = Subset(full_ds, list(range(n - n_val, n)))
        if stage in (None, "test"):
            self.test_set = LOBDataset(self.test_data, self.task_type, self.data_cfg)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch,
            shuffle=True,
            persistent_workers=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch,
            shuffle=False,
            persistent_workers=True,
            num_workers=self.num_workers,
        )
