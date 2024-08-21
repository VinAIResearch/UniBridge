import argparse

import torch
import wandb
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from UniBridgeSrc import LitPOSAdapter, POSAdapterDataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the subset dataset to train POS tagging")
    parser.add_argument("--subset", type=str)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epoch", type=int, default=10)
    args = parser.parse_args()

    sub_data = args.subset
    lr = args.lr
    epochs = args.epoch

    torch.set_float32_matmul_precision("high")

    logger.info(str(args))

    seed = seed_everything(42)
    lang = sub_data.split("_")[0]
    wandb_logger = WandbLogger(project="adapter", name=f"pos_{lang}_{lr}_{epochs}e", offline=True)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    dataloader_config = {
        "pretrained_ck": "bert-base-multilingual-cased",
        "ls_sub_data": sub_data,
        "max_length": 510,
    }
    pos_dataloader = POSAdapterDataLoader(**dataloader_config)
    [train_dataloader, valid_dataloader] = pos_dataloader.get_dataloader(batch_size=16, types=["train", "validation"])

    model_config = {
        "pretrained_ck": "bert-base-multilingual-cased",
        "lang_adapter_ckpt": f"src_ckpt/lang/{lang}/adapter",
        "lr": lr,
        "model": "mbert",
        "lang": lang,
    }

    lit_posadapter = LitPOSAdapter(**model_config)

    # train model
    trainer = Trainer(max_epochs=epochs, devices=[0], accelerator="gpu", logger=wandb_logger, callbacks=[lr_monitor])
    trainer.fit(model=lit_posadapter, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    lit_posadapter.export_model("src_ckpt/pos")

    wandb.finish()
