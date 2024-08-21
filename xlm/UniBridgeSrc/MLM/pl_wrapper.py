import lightning.pytorch as pl
import torch
import torch.nn as nn

from .configuration import MLMAdapterConfig
from .model import MLMAdapterModel


class LitMLMAdapter(pl.LightningModule):
    def __init__(self, pretrained_ck: str, lr: float, model: str, lang: str, pretrained_mlm_adapter: str = ""):
        super(LitMLMAdapter, self).__init__()
        config = MLMAdapterConfig.from_pretrained(
            pretrained_ck,
            pretrained_ck=pretrained_ck,
            model=model,
            lang=lang,
            pretrained_mlm_adapter=pretrained_mlm_adapter,
        )
        self.model = MLMAdapterModel(config)
        self.num_labels = config.vocab_size
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.save_hyperparameters()

    def export_model(self, path):
        self.model.save_pretrained(path)

    def training_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        logits = self.model(**batch)
        loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))
        self.log("train/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        logits = self.model(**batch)
        loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))
        self.log("valid/loss", loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=2, factor=0.1, threshold=0.1
            ),  # threshold=0.01
            "monitor": "valid/loss",
            "interval": "epoch",
            "frequency": 1,
            "name": "lr_monitor",
        }
        return [optimizer], [lr_scheduler]
