import glob
import os.path
import sys

import yaml
import logging
import argparse
from datetime import datetime
from pathlib import Path
import shutil

import torch.autograd
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from inkler.inkler import Inkler
from inkler.book_entry import BookDataset
from inkler.utils import get_git_revision_short_hash

logger = logging.getLogger(__name__)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml_path",
        required=True, type=str,
        help="Path to training yaml file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for RNG"
    )
    parser.add_argument(
        "--detect-anomaly",
        action="store_true",
        help="Enable torch.autograd.set_detect_anomaly for debugging NaN/Inf grads. "
             "Adds significant backward-pass overhead, so keep off for normal training runs."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to model checkpoint"
    )

    return parser.parse_args(args)

def main(args):
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    cnsl = logging.StreamHandler()
    logging.basicConfig(
        level=logging.INFO,  #todo flag
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%y-%m-%d %H:%M",
        handlers=[cnsl]
    )

    model_path = args.model_path
    train_path = args.yaml_path
    if not os.path.isfile(train_path):
        raise FileNotFoundError("Training yaml file not found")

    with open(train_path, "r") as yaml_file:
        train_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)

    seed = args.seed
    if seed is not None:
        seed_everything(seed, workers=True)

    # setup logging dir
    log_dir = train_dict["train"].get("logging_dir", None)
    name = train_dict["name"]
    if log_dir is not None:
        # setup log dir
        log_dir = os.path.join(log_dir, f"{name}_{datetime.today().strftime('%y%m%d')}")
        if os.path.isdir(log_dir):
            log_dir += f"_{len(glob.glob(f"{log_dir}*")) + 1}"
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        # copy setup yaml file
        shutil.copyfile(train_path, f"{log_dir}/{Path(train_path).name}")

        # add file logger
        file_log = logging.FileHandler(f"{log_dir}/train.log", mode="w")
        file_log.setLevel(logging.DEBUG)
        file_log.setFormatter(
            logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
        )
        main_log = logging.getLogger()
        main_log.addHandler(file_log)

        logger.info(f"Created logging directory at {log_dir}")

    sha = get_git_revision_short_hash()
    logger.info(f"Current git hash: {sha}")

    if "logging_dir" in train_dict["train"]:
        del train_dict["train"]["logging_dir"]
    if isinstance(log_dir, str):
        log_dir = os.path.join(log_dir, name)
        checkpoint_dir = os.path.join(log_dir, "checkpoints")
    else:
        checkpoint_dir = "lightning_logs/checkpoints"

    logger.info("Loading book dataset...")
    dataloader = BookDataset(
        seq_len=int(train_dict["model"]["seq_len"]),
        in_dim=int(train_dict["model"]["d_model"]),
        **train_dict["train_data"]
    )

    val_dataloader = BookDataset(
        seq_len=int(train_dict["model"]["seq_len"]),
        in_dim=int(train_dict["model"]["d_model"]),
        **train_dict["val_data"]
    )
    logger.info(f"Loaded training set with {dataloader.n_books} entries")

    n_pretrained_epochs = 0
    if model_path is None:
        model = Inkler(
            lr=train_dict["train"]["lr"],
            optim_params=train_dict["train"].get("optim_params", None),
            scheduler_params=train_dict["train"].get("scheduler_params", None),
            warmup_params=train_dict["train"].get("warmup_params", None),
            **train_dict["model"]
        )
        logger.info(f"Created new Inkler")
    else:
        try:
            n_pretrained_epochs = int(model_path.split("-")[1].replace("epoch=", ""))
        except ValueError:
            n_pretrained_epochs = 0
        model = Inkler.load_from_checkpoint(
            model_path,
            lr=train_dict["train"]["lr"],
            optim_params=train_dict["train"].get("optim_params", None),
            scheduler_params=train_dict["train"].get("scheduler_params", None),
            warmup_params=train_dict["train"].get("warmup_params", None),
            **train_dict["model"]
        ).train()
        logger.info(f"Loaded Inkler from checkpoint at path {model_path}")


    for param in ["lr", "optim_params", "scheduler_params", "warmup_params"]:
        if param in train_dict["train"]:
            del train_dict["train"][param]

    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        **train_dict["train"]["checkpoint"] if "checkpoint" in train_dict["train"] else {},
        dirpath=checkpoint_dir,
        filename=f"{name}-{{epoch:02d}}-{{{train_dict["train"]["checkpoint"]["monitor"]}:.2f}}",
    )
    if "checkpoint" in train_dict["train"]:
        del train_dict["train"]["checkpoint"]

    trainer = Trainer(
        **train_dict["train"],
        logger=CSVLogger(save_dir=log_dir if log_dir is not None else "lightning_logs/"),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            checkpoint_callback
        ],
    )
    trainer.fit_loop.epoch_progress.current.processed = n_pretrained_epochs
    trainer.fit(model=model, train_dataloaders=dataloader, val_dataloaders=val_dataloader)



if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))

