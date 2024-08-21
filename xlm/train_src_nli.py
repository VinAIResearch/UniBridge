import argparse

import torch
import wandb
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
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
    wandb_logger = WandbLogger(project="nli_adapter_xlm", name=f"{lang}", offline=True)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    dataloader_config = {
        "pretrained_ck": "xlm-roberta-base",
        "lang": lang,
        "max_length": 510,
    }
    nli_dataloader = TrainNLIAdapterDataLoader(**dataloader_config)
    [train_dataloader, valid_dataloader] = nli_dataloader.get_dataloader(batch_size=32, types=["train", "validation"])

    model_config = {
        "pretrained_ck": "xlm-roberta-base",
        "lang_adapter_ckpt": f"src_ckpt/lang/{lang}/adapter",
        "lr": lr,
        "model": "xlm-r",
        "lang": lang,
    }

    lit_posadapter = LitNLIAdapter(**model_config)

    # train model
    trainer = Trainer(max_epochs=epochs, devices=[0], accelerator="gpu", logger=wandb_logger, callbacks=[lr_monitor])
    trainer.fit(model=lit_posadapter, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    lit_posadapter.export_model("src_ckpt/nli")

    wandb.finish()
