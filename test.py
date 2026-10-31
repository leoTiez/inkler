import sys
import os.path

import pandas as pd
import yaml
import logging
import argparse
from pathlib import Path

import torch.autograd
from tqdm import tqdm

from inkler.inkler import Inkler
from inkler.book_entry import BookDataset

logger = logging.getLogger(__name__)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml_path",
        required=True,
        type=str,
        help="Path to training yaml file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )

    return parser.parse_args(args)

def main(args):
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

    model = Inkler.load_from_checkpoint(
        model_path,
        lr=train_dict["train"]["lr"],
        optim_params=train_dict["train"].get("optim_params", None),
        scheduler_params=train_dict["train"].get("scheduler_params", None),
        warmup_params=train_dict["train"].get("warmup_params", None),
        **train_dict["model"]
    ).eval()
    logger.info(f"Loaded Inkler from checkpoint at path {model_path}")

    out_dir = Path(model_path).parent.parent / "test_result"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading book test dataset...")
    train_dict["test_data"]["batch_size"] = 1
    test_dataloader = BookDataset(
        seq_len=int(train_dict["model"]["seq_len"]),
        in_dim=int(train_dict["model"]["d_model"]),
        **train_dict["test_data"]
    )
    logger.info(f"Loaded training set with {test_dataloader.n_books} entries")

    logger.info("Start testing...")
    with torch.no_grad():
        loss_l = []
        for batch in tqdm(test_dataloader):
            result_dict = model.forward(
                x_sentence=batch["texts"],
                hierarchy=batch["hierarchy"],
                attention_mask=batch["masks"]
            )
            loss_dict = model.loss(
                fwd_out=result_dict,
                true_score=batch["scores"],
                true_genre=batch["genres"],
                postits=batch.get("postits", None),
            )
            loss = (
                    loss_dict["recon_loss"] * model.recon_w
                    + loss_dict["decoder_loss"] * model.decode_w
                    + loss_dict["topology_loss"] * model.topology_w
            )
            loss_l.append((batch["authors"][0], batch["titles"][0], loss.item()))

    logger.info(f"Saving loss test results {out_dir}/test.csv")
    pd.DataFrame(loss_l).to_csv(f"{out_dir}/test_loss.csv", header=False, index=False)

if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))

