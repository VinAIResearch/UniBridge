import argparse

import torch
import wandb
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from UniBridgeSrc import LitNLIAdapter, TrainNLIAdapterDataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the dataset to train NLI")
    parser.add_argument("--lang", type=str)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epoch", type=int, default=10)
    args = parser.parse_args()

    lang = args.lang
    lr = args.lr
    epochs = args.epoch

    torch.set_float32_matmul_precision("high")

    seed = seed_everything(42)
    logger.info(str(args))
    wandb_logger = WandbLogger(project="adapter", name=f"nli_{lang}_{lr}_{epochs}e", offline=True)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    dataloader_config = {
        "pretrained_ck": "bert-base-multilingual-cased",
        "lang": lang,
        "max_length": 510,
    }
    nli_dataloader = TrainNLIAdapterDataLoader(**dataloader_config)
    [train_dataloader, valid_dataloader] = nli_dataloader.get_dataloader(batch_size=16, types=["train", "validation"])

    model_config = {
        "pretrained_ck": "bert-base-multilingual-cased",
        "lang_adapter_ckpt": f"src_ckpt/lang/{lang}/adapter",
        "lr": lr,
        "model": "xlm-r",
        "lang": lang,
    }

    lit_nliadapter = LitNLIAdapter(**model_config)

    # train model
    trainer = Trainer(max_epochs=epochs, devices=[0], accelerator="gpu", logger=wandb_logger, callbacks=[lr_monitor])
    trainer.fit(model=lit_nliadapter, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    lit_nliadapter.export_model("src_ckpt/nli")

    wandb.finish()
