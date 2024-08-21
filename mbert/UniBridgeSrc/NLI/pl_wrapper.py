import evaluate
import lightning.pytorch as pl
import torch
import torch.nn as nn

from .configuration import NLIAdapterConfig
from .label_converter import LabelConverter
from .model import NLIAdapterModel


class LitNLIAdapter(pl.LightningModule):
    def __init__(self, pretrained_ck: str, lang_adapter_ckpt: str, lr: float, model: str, lang: str):
        super(LitNLIAdapter, self).__init__()
        label_converter = LabelConverter()
        self.id2label = label_converter.id2label
        config = NLIAdapterConfig.from_pretrained(
            pretrained_ck,
            pretrained_ck=pretrained_ck,
            lang_adapter_ckpt=lang_adapter_ckpt,
            model=model,
            lang=lang,
        )
        self.model = NLIAdapterModel(config)
        self.model_type = model
        self.lang = lang
        self.num_labels = config.num_labels
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.valid_metric = evaluate.load("metrics/nli/accuracy.py")
        self.save_hyperparameters()

    def __postprocess(self, predictions, labels):
        predictions = predictions.detach().cpu().clone().numpy()
        labels = labels.detach().cpu().clone().numpy()

        return predictions, labels

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
        predictions = logits.argmax(dim=-1)

        decoded_preds, decoded_labels = self.__postprocess(predictions, labels)
        self.valid_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    def on_validation_epoch_end(self):
        results = self.valid_metric.compute()
        self.log("valid/accuracy", results["accuracy"], on_epoch=True, on_step=False, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, threshold=0.1, mode="max"),
            "monitor": "valid/accuracy",
            "interval": "epoch",
            "frequency": 1,
            "name": "lr_monitor",
        }
        return [optimizer], [lr_scheduler]
