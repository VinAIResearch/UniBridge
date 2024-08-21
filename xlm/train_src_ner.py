import argparse

import torch
import wandb
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from UniBridgeSrc import LitNERAdapter, NERAdapterDataLoader


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

    # dataloader
    dataloader_config = {
        "pretrained_ck": "xlm-roberta-base",  # 'bert-base-multilingual-cased', 'xlm-roberta-base'
        "model": "xlm-r",  # mbert, xlm-r
        "lang": lang,
        "max_length": 510,
    }
    ner_dataloader = NERAdapterDataLoader(**dataloader_config)
    [train_dataloader, valid_dataloader] = ner_dataloader.get_dataloader(batch_size=32, types=["train", "test"])

    wandb_logger = WandbLogger(project="madx_adapter", name=f"ner_{lang}_{lr}_{epochs}e", offline=True)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    model_config = {
        "pretrained_ck": "xlm-roberta-base",  # 'bert-base-multilingual-cased', 'xlm-roberta-base'
        "model": "xlm-r",  # mbert, xlm-r
        "lr": lr,  # 1e-5,2e-5,5e-5,1e-4,2e-4,5e-4
        "lang": lang,
        "lang_adapter_ckpt": f"src_ckpt/lang/{lang}/adapter",
    }
    lit_neradapter = LitNERAdapter(**model_config)

    # train model
    trainer = Trainer(
        max_epochs=epochs, devices=[0], accelerator="gpu", logger=wandb_logger, callbacks=[lr_monitor]
    )  # , deterministic=True, precision='bf16')#, strategy="ddp")
    trainer.fit(model=lit_neradapter, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    lit_neradapter.export_model("src_ckpt/ner")

    wandb.finish()
