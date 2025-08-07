import torch.nn as nn, torch, pytorch_lightning as pl
import torch.nn.functional as F


class DeepLOBLightning(pl.LightningModule):
    def __init__(
        self,
        input_width,
        input_size,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        lr,
        neg_slope,
        hidden_size,
        lr_reduce_patience,
        task_type,
        monitor_metric,
        mode,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.valid_preds: list[torch.Tensor] = []
        self.valid_labels: list[torch.Tensor] = []

        self.act = nn.LeakyReLU(self.hparams.neg_slope)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )  # -> (B,16,100,input_width // 2)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, stride=stride
        )  # -> (B,16,100,input_width // 4)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(1, self.hparams.input_width // 4)
        )  # -> (B,16,100,1)

        # Inception 分支
        def branch(k):  # k x 1 卷积分支
            return nn.Sequential(
                nn.Conv2d(16, 8, (1, 1)),
                nn.Conv2d(8, 8, (k, 1), padding="same"),
                self.act,
            )

        self.branch1 = nn.Conv2d(16, 8, (1, 1))
        self.branch3, self.branch10, self.branch20 = branch(3), branch(10), branch(20)
        self.branchP = nn.Sequential(
            nn.MaxPool2d((3, 1), 1, (1, 0)), nn.Conv2d(16, 8, (1, 1))
        )

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )

        if task_type == "classification":
            self.out_dim = 3
            self.loss_fn = nn.CrossEntropyLoss()
        elif task_type == "regression":
            self.out_dim = 1
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        self.fc = nn.Linear(hidden_size, self.out_dim)
        self.apply(self._init_weights)
        self.loss = nn.CrossEntropyLoss()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(
                m.weight,
                a=self.hparams.neg_slope if hasattr(self.hparams, "neg_slope") else 0,
                nonlinearity="leaky_relu",
            )
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):  # x (B,100,40)
        x = x.unsqueeze(1)  # -> (B,1,100,40)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))  # (B,16,100,1)

        B, C, T, W = x.shape  # W=1
        b1 = self.act(self.branch1(x))
        b3 = self.branch3(x)
        b10 = self.branch10(x)
        b20 = self.branch20(x)
        bp = self.act(self.branchP(x))
        x = torch.cat([b1, b3, b10, b20, bp], dim=1)  # (B,40,100,1)
        x = x.squeeze(-1).permute(0, 2, 1)  # (B,40,100)

        lstm_out, _ = self.lstm(x)  # (100,B,64)
        logits = self.fc(lstm_out[:, -1, :])  # (B, out_dim)
        return logits

    def training_step(self, batch, batch_idx):
        X, y = batch
        preds = self(X)
        if self.hparams.task_type == "classification":
            loss = self.loss_fn(preds, y)
        else:
            loss = self.loss_fn(preds.squeeze(-1), y.float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        preds = self(X)
        if self.hparams.task_type == "classification":
            acc = (preds.argmax(dim=1) == y).float().mean()
            self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        else:
            preds = preds.squeeze(-1)
            labels = y.float()
            mse = F.mse_loss(preds.squeeze(-1), y.float())
            self.log("val_mse", mse, on_step=False, on_epoch=True, prog_bar=True)
            self.valid_preds.append(preds.detach().cpu())
            self.valid_labels.append(labels.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        all_preds  = torch.cat(self.valid_preds, dim=0)
        all_labels = torch.cat(self.valid_labels, dim=0)

        nonzero = (all_preds != 0).sum().item()
        total   = all_preds.numel()
        print(f"[val epoch] non-zero preds = {nonzero}/{total}")

        ic = torch.corrcoef(torch.stack([all_preds, all_labels], dim=0))[0, 1]
        self.log("val_ic", ic, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.hparams.mode,
            factor=0.5,
            patience=self.hparams.lr_reduce_patience,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.hparams.monitor_metric,  # 以 validation accuracy 为触发指标
                "interval": "epoch",
                "frequency": 1,
            },
        }
