import argparse

import torch
import wandb
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from UniBridge import LitUniBridge, UniBridgeDataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the language to train adapter")
    parser.add_argument("--lang", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mlm_epoch", type=int, default=15)
    args = parser.parse_args()

    lang = args.lang
    s = args.seed
    lr = args.lr
    seed = seed_everything(s, workers=True)
    epochs = args.mlm_epoch  # 50, 100, 150, 250
    torch.set_float32_matmul_precision("high")

    logger.info(str(args))
    out_dict = {}
    data_config = {
        "tok_pretrained_ck": f"tokenizer_ckpt/sptok/{lang}",
        "lang": lang,
        "prob": 0.15,
        "test_ratio": 0.01,
        "max_length": 510,
        "chunk_size": 128,
        "smart_chunk_size": True,
        "out_dict": out_dict,
    }
    mlm_dataloader = UniBridgeDataLoader(**data_config)
    [train_dataloader, valid_dataloader] = mlm_dataloader.get_dataloader(batch_size=32, types=["train", "test"])

    model_config = {
        "model_pretrained_ck": "bert-base-multilingual-cased",
        "lr": lr,
        "lang": lang,
        "model": "mbert",
        "pretrained_mlm_adapter": "",
        "embed_pretrained_ckpt": f"init_embed_ckpt/{lang}/embed_model",
    }
    lit_UniBridge = LitUniBridge(**model_config)

    wandb_logger = WandbLogger(project="UniBridge_mlm_mbert", name=f"{lang}_{lr}_{epochs}", offline=True)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # train model
    trainer = Trainer(max_epochs=epochs, devices=[0], accelerator="gpu", logger=wandb_logger, callbacks=[lr_monitor])
    trainer.fit(model=lit_UniBridge, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    lit_UniBridge.export_model("UniBridge_ckpt/beta-0.5")

    wandb.finish()
