from typing import Dict

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM
from UniBridge.MLM import UniBridgeConfig, UniBridgeModel


class LitUniBridge(pl.LightningModule):
    def __init__(
        self,
        model_pretrained_ck: str,
        lr: float,
        model: str,
        lang: str,
        num_training_steps: int,
        ratio: float,
        pretrained_mlm_adapter: str = "",
        embed_pretrained_ckpt: str = "",
    ):
        super(LitUniBridge, self).__init__()
        self.base_model = AutoModelForMaskedLM.from_pretrained(model_pretrained_ck, output_hidden_states=True)
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        config = UniBridgeConfig.from_pretrained(
            model_pretrained_ck,
            pretrained_ck=model_pretrained_ck,
            model=model,
            lang=lang,
            pretrained_mlm_adapter=pretrained_mlm_adapter,
            embed_pretrained_ckpt=embed_pretrained_ckpt,
        )
        self.model = UniBridgeModel(config)
        self.ratio = ratio
        self.num_training_steps = num_training_steps
        self.num_labels = config.vocab_size
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.save_hyperparameters()

    def export_model(self, path):
        self.model.save_pretrained(path)

    def unlink(self):
        return self.model

    def __mean_pooling(self, model_output: Dict, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def training_step(self, batch, batch_idx):
        labels = batch.pop("labels")

        with torch.inference_mode():
            out = self.base_model(**batch)
            base_hidden_state = F.softmax(self.__mean_pooling(out.hidden_states[-1], batch["attention_mask"]), dim=-1)

        out = self.model(**batch)
        logits = out.logits
        last_hidden_state = F.softmax(self.__mean_pooling(out.hidden_states[-1], batch["attention_mask"]), dim=-1)
        mlm_loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))
        kl_loss = (-1) * F.kl_div(last_hidden_state, base_hidden_state.clone(), reduction="batchmean")
        loss = mlm_loss + kl_loss
        self.log("train/loss", loss, sync_dist=True)
        self.log("train/mlm_loss", mlm_loss, sync_dist=True)
        self.log("train/kl_loss", kl_loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        logits = self.model(**batch).logits
        loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))
        self.log("valid/loss", loss, sync_dist=True)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)  # , weight_decay=0.01)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=3, factor=0.6, threshold=0.1
            ),  # factor=0.6
            "monitor": "valid/loss",
            "interval": "epoch",
            "frequency": 1,
            "name": "lr_monitor",
        }
        return [optimizer], [lr_scheduler]
