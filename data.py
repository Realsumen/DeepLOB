import numpy as np, torch.nn as nn, pandas as pd, torch, pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from typing import List, Union
from torch.utils.data import Subset


class LOBDataset(Dataset):
    """
    支持传入单个 pd.DataFrame，或 List[pd.DataFrame]。
    每条样本由 (df_id, k) 唯一确定，其中 k 是各自 df 的行号。
    """

    def __init__(self, data, data_cfg: dict):
        super().__init__()
        self.seq_len = data_cfg["seq_len"]
        self.horizon = data_cfg["horizon"]
        self.alpha = data_cfg["alpha"]
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
        else:
            past = mid_arr[k]
        future = mid_arr[k + 1 : k + 1 + self.horizon].mean()

        pct = (future - past) / past
        label = 2 if pct > self.alpha else (0 if pct < -self.alpha else 1)

        return torch.from_numpy(x_window), torch.tensor(label, dtype=torch.long)


class LOBDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: Union[pd.DataFrame, List[pd.DataFrame]],
        batch: int,
        val_ratio: float,
        data_cfg: dict,
        random_split: bool = False,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data, self.batch, self.val_ratio = data, batch, val_ratio
        self.data_cfg = data_cfg
        self.random_split = random_split
        self.num_workers = num_workers

    def setup(self, stage=None):
        ds = LOBDataset(self.data, self.data_cfg)

        if self.random_split:
            n_val = int(len(ds) * self.val_ratio)
            self.train_set, self.val_set = torch.utils.data.random_split(
                ds, [len(ds) - n_val, n_val]
            )
        else:
            n = len(ds)
            n_val = int(n * self.val_ratio)
            train_indices = list(range(0, n - n_val))
            val_indices = list(range(n - n_val, n))

            self.train_set = Subset(ds, train_indices)
            self.val_set = Subset(ds, val_indices)

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
