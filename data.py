import numpy as np, torch.nn as nn, pandas as pd, torch, pytorch_lightning as pl
from typing import Optional, Tuple, List, Union, Dict
from torch.utils.data import Dataset, DataLoader


class LOBDataset(Dataset):

    def __init__(
        self,
        data: List[pd.DataFrame],
        task_type: str,
        data_cfg: dict,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.task_type = task_type
        self.seq_len = data_cfg["seq_len"]
        self.horizon = data_cfg["horizon"]
        self.alpha = data_cfg["alpha"]
        self.multiplier = data_cfg["multiplier"]
        self.features = data_cfg["feature_order"]
        self.use_rolling_mean = data_cfg["use_rolling_mean"]
        self.eps = eps

        if isinstance(data, pd.DataFrame):
            data = [data]

        self.X_list, self.mid_list = [], []
        self.sample_map = []  # [(df_id, k), ...]

        for df_id, df in enumerate(data):
            # 注意：此处仅做 ffill，标准化已在 DataModule 完成
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

        label = (future - past) / (past + self.eps) * self.multiplier

        if self.task_type == "regression":
            return torch.from_numpy(x_window), torch.tensor(label, dtype=torch.float32)
        elif self.task_type == "classification":
            lab = 2 if label > self.alpha else (0 if label < -self.alpha else 1)
            return torch.from_numpy(x_window), torch.tensor(lab, dtype=torch.long)
        else:
            raise RuntimeError(f"{self.task_type} is not support.")


class LOBDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dm_cfg: dict,
        data_cfg: dict,
        train_data: Union[pd.DataFrame, List[pd.DataFrame]],
        test_data: Union[pd.DataFrame, List[pd.DataFrame]],
        val_data: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
    ):
        super().__init__()
        self.batch = dm_cfg["batch_size"]
        self.val_ratio = dm_cfg["val_ratio"]
        self.num_workers = dm_cfg.get("num_workers", 4)
        self.symbol_field = dm_cfg["symbol_field"]
        self.do_std = dm_cfg.get("do_std", True)
        self.resplit_from_pool = dm_cfg.get("resplit_from_pool", False)
        self.seed = dm_cfg.get("seed", None)
        self.task_type = dm_cfg.get("task_type", "regression")
        self.norm_eps = dm_cfg.get("norm_eps", 1e-8)

        self.data_cfg = data_cfg
        self.train_data, self.val_data, self.test_data = train_data, val_data, test_data

        self.per_symbol_mu: Optional[Dict[Union[str, int, float], np.ndarray]] = None
        self.per_symbol_std: Optional[Dict[Union[str, int, float], np.ndarray]] = None
        self.global_mu: Optional[np.ndarray] = None
        self.global_std: Optional[np.ndarray] = None

        self._normalized_frames = set()  # 防止重复标准化（用 id(df) 标记）

        self.train_set = self.val_set = self.test_set = None

    @staticmethod
    def _to_list(data):
        return data if isinstance(data, list) else [data]

    @staticmethod
    def _merge_to_list(a, b):
        la = a if isinstance(a, list) else ([a] if a is not None else [])
        lb = b if isinstance(b, list) else ([b] if b is not None else [])
        return la + lb

    # ---------- 统计函数 ----------

    def _accumulate_per_symbol_stats(
        self, frames: List[pd.DataFrame]
    ) -> Tuple[Dict, Dict, Dict]:
        """
        返回：sum_dict, sumsq_dict, count_dict（key= symbol, value= np.ndarray 或 int）
        按 symbol_field 聚合特征列的 sum / sumsq / count。
        使用 ffill 后 dropna，避免 NaN 污染统计。
        """
        feats = self.data_cfg["feature_order"]
        F = len(feats)
        s_col = self.symbol_field

        sum_dict: Dict[Union[str, int, float], np.ndarray] = {}
        sumsq_dict: Dict[Union[str, int, float], np.ndarray] = {}
        count_dict: Dict[Union[str, int, float], int] = {}

        for df in frames:
            if s_col not in df.columns:
                symbol_values = [f"__single_{id(df)}__"]
                df_local = df.copy()
                df_local[s_col] = symbol_values[0]
            else:
                df_local = df

            for sym in df_local[s_col].dropna().unique():
                mask = df_local[s_col] == sym
                sub = df_local.loc[mask, feats]

                sub_filled = sub.ffill().dropna()
                if sub_filled.empty:
                    continue
                arr = sub_filled.to_numpy(dtype=np.float64)

                s = arr.sum(axis=0, dtype=np.float64)
                ss = np.square(arr, dtype=np.float64).sum(axis=0)

                if sym not in sum_dict:
                    sum_dict[sym] = s
                    sumsq_dict[sym] = ss
                    count_dict[sym] = arr.shape[0]
                else:
                    sum_dict[sym] += s
                    sumsq_dict[sym] += ss
                    count_dict[sym] += arr.shape[0]

        return sum_dict, sumsq_dict, count_dict

    def _calc_global_mu_sigma_from_frames(
        self, frames: List[pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray]:
        feats = self.data_cfg["feature_order"]
        F = len(feats)
        total = np.zeros(F, np.float64)
        total_sq = np.zeros(F, np.float64)
        count = 0
        for df in frames:
            arr = df[feats].ffill().bfill().dropna().to_numpy(np.float64)
            if arr.size == 0:
                continue
            total += arr.sum(0, dtype=np.float64)
            total_sq += np.square(arr, dtype=np.float64).sum(0)
            count += arr.shape[0]
        if count == 0:
            mean = np.zeros(F, np.float64)
            std = np.ones(F, np.float64)
        else:
            mean = total / count
            var = np.maximum(total_sq / count - mean * mean, 0.0)
            std = np.sqrt(var)
            std[std == 0] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)

    def _finalize_symbol_mu_std(
        self,
        sums: Dict,
        sumsqs: Dict,
        counts: Dict,
        F: int,
    ) -> Tuple[Dict, Dict]:
        mu_d, std_d = {}, {}
        for sym, c in counts.items():
            if c <= 0:
                mu = np.zeros(F, np.float64)
                std = np.ones(F, np.float64)
            else:
                mu = sums[sym] / c
                var = np.maximum(sumsqs[sym] / c - mu * mu, 0.0)
                std = np.sqrt(var)
                std[std == 0] = 1.0
            mu_d[sym] = mu.astype(np.float32)
            std_d[sym] = std.astype(np.float32)
        return mu_d, std_d

    def _standardize_frames_inplace(
        self,
        frames: List[pd.DataFrame],
        per_symbol_mu: Dict,
        per_symbol_std: Dict,
        global_mu: np.ndarray,
        global_std: np.ndarray,
        eps: float,
    ) -> None:
        feats = self.data_cfg["feature_order"]
        s_col = self.symbol_field

        for df in frames:
            if id(df) in self._normalized_frames:
                continue  # 已处理，避免重复标准化

            if s_col not in df.columns:
                X = df[feats].ffill().to_numpy(np.float32)
                X = (X - global_mu) / (global_std + eps)
                df.loc[:, feats] = X.astype(np.float32)
                self._normalized_frames.add(id(df))
                continue

            symbols = df[s_col].astype(object).values  # 保持通用类型
            # 为避免 groupby 带来的额外开销，用掩码循环
            for sym in pd.unique(symbols):
                mask = df[s_col] == sym
                sub = df.loc[mask, feats].ffill()

                mu = per_symbol_mu.get(sym, global_mu)
                std = per_symbol_std.get(sym, global_std)

                X = sub.to_numpy(np.float32)
                X = (X - mu) / (std + eps)

                df.loc[mask, feats] = X.astype(np.float32)

            self._normalized_frames.add(id(df))

    def setup(self, stage: Optional[str] = None):
        feats = self.data_cfg["feature_order"]
        F = len(feats)

        # --- fit 阶段：切分并计算训练统计 ---
        if stage in (None, "fit"):
            if self.resplit_from_pool:
                pool = self._merge_to_list(self.train_data, self.val_data)
                if len(pool) == 0:
                    raise ValueError(
                        "resplit=True 需要 train_data 或 val_data 至少一个非空。"
                    )

                g = torch.Generator()
                if self.seed is not None:
                    g.manual_seed(int(self.seed))
                perm = torch.randperm(len(pool), generator=g).tolist()
                n_val = max(1, int(len(pool) * self.val_ratio))
                n_val = min(n_val, len(pool) - 1)  # 保证两边都有
                val_idx = set(perm[:n_val])

                train_frames = [pool[i] for i in range(len(pool)) if i not in val_idx]
                val_frames = [pool[i] for i in range(len(pool)) if i in val_idx]
            else:
                if self.val_data is None:
                    raise ValueError("resplit=False 时必须提供 val_data。")
                train_frames = self._to_list(self.train_data)
                val_frames = self._to_list(self.val_data)

            # 计算 per-symbol & global 统计（仅基于训练集）
            if self.do_std:
                sums, sumsqs, counts = self._accumulate_per_symbol_stats(train_frames)
                self.per_symbol_mu, self.per_symbol_std = self._finalize_symbol_mu_std(
                    sums, sumsqs, counts, F
                )
                self.global_mu, self.global_std = (
                    self._calc_global_mu_sigma_from_frames(train_frames)
                )

                # 对 train/val 做就地标准化
                self._standardize_frames_inplace(
                    train_frames,
                    self.per_symbol_mu,
                    self.per_symbol_std,
                    self.global_mu,
                    self.global_std,
                    self.norm_eps,
                )
                self._standardize_frames_inplace(
                    val_frames,
                    self.per_symbol_mu,
                    self.per_symbol_std,
                    self.global_mu,
                    self.global_std,
                    self.norm_eps,
                )

            # 构造数据集（此时不再传 mu/std，__getitem__ 不做标准化）
            self.train_set = LOBDataset(
                train_frames,
                self.task_type,
                self.data_cfg,
                eps=self.norm_eps,
            )
            self.val_set = LOBDataset(
                val_frames,
                self.task_type,
                self.data_cfg,
                eps=self.norm_eps,
            )

        if stage in (None, "test"):
            test_frames = self._to_list(self.test_data)

            if self.do_std:
                if self.per_symbol_mu is None or self.per_symbol_std is None:
                    sums, sumsqs, counts = self._accumulate_per_symbol_stats(
                        test_frames
                    )
                    self.per_symbol_mu, self.per_symbol_std = (
                        self._finalize_symbol_mu_std(sums, sumsqs, counts, F)
                    )
                if self.global_mu is None or self.global_std is None:
                    self.global_mu, self.global_std = (
                        self._calc_global_mu_sigma_from_frames(test_frames)
                    )

                self._standardize_frames_inplace(
                    test_frames,
                    self.per_symbol_mu,
                    self.per_symbol_std,
                    self.global_mu,
                    self.global_std,
                    self.norm_eps,
                )

            self.test_set = LOBDataset(
                test_frames,
                self.task_type,
                self.data_cfg,
                eps=self.norm_eps,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0),
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0),
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0),
            pin_memory=torch.cuda.is_available(),
        )
