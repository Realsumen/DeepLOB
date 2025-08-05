from sklearn.metrics import confusion_matrix, classification_report
import torch

def evaluate_model_on_val(model: torch.nn.Module, datamodule: torch.utils.data.DataLoader):
    
    all_preds, all_labels = [], []
    device = model.device if hasattr(model, "device") else torch.device("cpu")

    with torch.no_grad():
        for X, y in datamodule.val_dataloader():
            X = X.to(device)
            logits = model(X)
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(y)

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    cm = confusion_matrix(labels, preds, labels=[0,1,2])
    print("混淆矩阵 (行=真值, 列=预测)：")
    print(cm)

    print("\n分类报告：")
    print(classification_report(
        labels, preds,
        labels=[0,1,2],
        target_names=["跌", "平", "涨"],
        digits=4
    ))
