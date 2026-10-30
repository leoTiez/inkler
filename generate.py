import sys
import os.path

import yaml
import logging
import argparse
from pathlib import Path
import shutil

import torch.autograd
from tqdm import tqdm

from inkler.inkler import Inkler
from inkler.book_entry import BookDataset
from inkler.stroke_topology import get_stroke_topology

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
DATASETS = ["train_data", "val_data", "test_data"]
LEVEL_NAMES = {
    -1: "sentence",
    0: "paragraph",
    1: "chapter",
    2: "book"
}

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
    parser.add_argument(
        "--n_points",
        type=int,
        default=64,
        help="Number of points per stroke"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Path to output directory"
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
    out_dir = args.out_dir
    n_points = args.n_points
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

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(train_path, os.path.join(out_dir, "config.yaml"))
    out_dir = os.path.join(out_dir, "glyphs")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        logger.info(f"Loading book dataset type {dataset}...")
        train_dict[dataset]["batch_size"] = 1
        dataloader = BookDataset(
            seq_len=int(train_dict["model"]["seq_len"]),
            in_dim=int(train_dict["model"]["d_model"]),
            **train_dict[dataset]
        )
        logger.info(f"Loaded book set with {dataloader.n_books} entries")

        logger.info("Start generating glyphs...")
        with torch.no_grad():
            for batch in tqdm(dataloader):
                result_dict = model.forward(
                    x_sentence=batch["texts"],
                    hierarchy=batch["hierarchy"],
                    attention_mask=batch["masks"]
                )

                val_glyphs = {key: val for key, val in result_dict.items() if "glyph" in key}
                for glyph_level, glyph in val_glyphs.items():
                    try:
                        level = int(glyph_level.replace("glyph_", ""))
                    except:
                        break
                    topology = get_stroke_topology(
                        n_strokes=model.n_strokes if model.consistent_hierarchy else model.n_strokes * 2 ** (level + 2),
                        n_connect=model.n_connect,
                        use_connector_stroke=True,
                    )
                    pg = model.character(glyph, topology=topology, n_points=n_points)
                    for i_gfd, glyph_for_drawing in enumerate(pg.transpose(0, 1)):
                        if not (~result_dict[glyph_level.replace("glyph_", "deplex_mask_")].bool())[0, i_gfd]:
                            logger.info(f"Finished drawing level {LEVEL_NAMES[level]}")
                            break
                        for i_glyph, rg in enumerate(glyph_for_drawing):
                            glyph_out =  f"{"_".join(batch["titles"][i_glyph].split(" "))}_BY_{
                                "_".join(batch["authors"][i_glyph].split(" "))}"
                            Path(os.path.join(out_dir, glyph_out)).mkdir(parents=True, exist_ok=True)
                            save_name = f"level_{LEVEL_NAMES[level]}_{i_gfd}.svg"
                            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                            ax.plot(*rg.permute(2, 1, 0).detach().cpu().numpy(), color="black")
                            ax.axis("off")
                            fig.tight_layout()
                            fig.savefig(f"{out_dir}/{glyph_out}/{save_name}")
                            plt.close(fig)
                        # clear
                        del glyph_for_drawing
                plt.close("all")


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))