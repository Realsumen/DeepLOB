from sklearn.metrics import confusion_matrix, classification_report
import pytorch_lightning as pl
import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


def evaluate_model(
    model: pl.LightningModule, datamodule: pl.LightningDataModule, dataset: str = "test"
):
    task_type = model.hparams.task_type
    all_preds, all_labels = [], []
    device = model.device if hasattr(model, "device") else torch.device("cpu")

    if dataset == "test":
        dataloader = datamodule.test_dataloader()
    elif dataset == "val":
        dataloader = datamodule.val_dataloader()
    elif dataset == "train":
        dataloader = datamodule.train_dataloader()
    else:
        raise RuntimeError("Invalid dataset")

    with torch.no_grad():

        for X, y in tqdm(dataloader):
            X = X.to(device)
            out = model(X)
            if task_type == "classification":
                preds = out.argmax(dim=1).cpu()
                labels = y.long()  # 确保整型
            else:
                preds = out.squeeze(-1).cpu()        # (B,)
                labels = y.float() 
            all_preds.append(preds)
            all_labels.append(labels)

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    if task_type == "classification":

        cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
        print("混淆矩阵 (行=真值, 列=预测)：")
        print(cm)

        print("\n分类报告：")
        print(
            classification_report(
                labels,
                preds,
                labels=[0, 1, 2],
                target_names=["跌", "平", "涨"],
                digits=4,
            )
        )

    elif task_type == "regression":

        ic_pearson, _ = pearsonr(preds, labels)
        ic_spearman, _ = spearmanr(preds, labels)

        print(f"Pearson IC = {ic_pearson:.4f}")
        print(f"Spearman IC = {ic_spearman:.4f}")

        lr = LinearRegression(fit_intercept=False)
        lr.fit(preds.reshape(-1, 1), labels)
        r2_no_intercept = lr.score(preds.reshape(-1, 1), labels)
        print(f"R^2 (no intercept, standard) = {r2_no_intercept:.4f}")

        num = np.dot(preds, labels) ** 2
        den = np.dot(preds, preds) * np.dot(labels, labels)
        r2_formula = num / den
        print(f"R^2 (no intercept, formula) = {r2_formula:.4f}")

        mse = mean_squared_error(labels, preds)
        print(f"MSE = {mse:.6f}")
