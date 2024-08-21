import argparse

import torch
import wandb
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from UniBridgeSrc import LitMLMAdapter, MLMAdapterDataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the language to train adapter")
    parser.add_argument("--lang", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epoch", type=int, default=21)
    args = parser.parse_args()

    lang = args.lang
    s = args.seed
    lr = args.lr
    seed = seed_everything(s, workers=True)
    epochs = args.epoch  # 50, 100, 150, 250
    torch.set_float32_matmul_precision("high")
    logger.info(str(args))

    # dataloader
    dataloader_config = {
        "pretrained_ck": "bert-base-multilingual-cased",
        "model": "mbert",
        "lang": lang,
        "prob": 0.15,
        "test_ratio": 0.01,
        "max_length": 510,
        "chunk_size": 384,
    }
    mlm_dataloader = MLMAdapterDataLoader(**dataloader_config)
    [train_dataloader, valid_dataloader] = mlm_dataloader.get_dataloader(batch_size=16, types=["train", "test"])

    wandb_logger = WandbLogger(project="adapter", name=f"mlm_{lang}_{lr}_{epochs}e", offline=True)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    model_config = {
        "pretrained_ck": "bert-base-multilingual-cased",
        "model": "mbert",
        "lr": lr,
        "lang": lang,
        "pretrained_mlm_adapter": "",
    }
    lit_mlmadapter = LitMLMAdapter(**model_config)

    # train model
    trainer = Trainer(max_epochs=epochs, devices=[0], accelerator="gpu", logger=wandb_logger, callbacks=[lr_monitor])
    trainer.fit(model=lit_mlmadapter, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    lit_mlmadapter.export_model("src_ckpt/lang")

    wandb.finish()
