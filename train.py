import os.path
import sys

import yaml
import logging
import argparse

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from inkler.inkler import Inkler
from inkler.book_entry import BookDataset

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

    return parser.parse_args(args)

def main(args):
    cnsl = logging.StreamHandler()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%y-%m-%d %H:%M",
        handlers=[cnsl]
    )

    train_path = args.yaml_path
    if not os.path.isfile(train_path):
        raise FileNotFoundError("Training yaml file not found")

    with open(train_path, "r") as yaml_file:
        train_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)

    seed = args.seed
    seed_everything(seed, workers=True)

    logger.info("Loading book dataset...")
    dataloader = BookDataset(
        seq_len=int(train_dict["model"]["seq_len"]),
        in_dim=int(train_dict["model"]["d_model"]),
        **train_dict["data"]
    )
    logger.info(f"Loaded training set with {len(dataloader)} entries")

    model = Inkler(
        lr=train_dict["train"]["lr"],
        optim_params=train_dict["train"].get("optim_params", None),
        scheduler_params=train_dict["train"].get("scheduler_params", None),
        **train_dict["model"]
    )

    del train_dict["train"]["lr"]
    del train_dict["train"]["optim_params"]
    del train_dict["train"]["scheduler_params"]
    log_dir = train_dict["train"].get("logging_dir", None)
    if "logging_dir" in train_dict["train"]:
        del train_dict["train"]["logging_dir"]
    trainer = Trainer(
        **train_dict["train"],
        logger=CSVLogger(save_dir=log_dir if log_dir is not None else "lightning_logs/"),
        callbacks=[LearningRateMonitor(logging_interval='step')]
    )
    trainer.fit(model=model, train_dataloaders=dataloader)



if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))

