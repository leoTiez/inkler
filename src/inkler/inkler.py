import logging
from typing import Callable, Tuple, Dict, Any, Optional
from pathlib import Path
import os
import math

import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from inkler import sdp
from inkler.constants import N_GENRES
from inkler import utils
from inkler.stroke_topology import get_stroke_topology
from inkler.suppl_inkler import InklerScore, InklerGenre, GlyphDecoder
from inkler.renderer import GlyphRenderer

from inkler.lexic.glyphmap import LexicCharacterStrokes, LexicInklerGlyph, lexic_contact_loss
from inkler.lexic.bezier import BezierCharacterStrokes, BezierInklerGlyph, bezier_contact_loss
from inkler.lexic.ode_glyph import ODECharacterStrokes, ODEInklerGlyph, ode_contact_loss
from inkler.lexic.woodlouse import WoodlouseCharacterStrokes, WoodlouseInklerGlyph, woodlouse_contact_loss

logger = logging.getLogger(__name__)


class Inkler(pl.LightningModule):
    def __init__(
            self,
            d_model: int = int(2**9),
            seq_len: int = int(2**15),
            n_heads: int = 16,
            d_ff: int = int(2**10),
            n_enc_blocks: int = 8,
            n_dec_blocks: int = 1,
            n_xca: int = 2,
            n_hierarchy: int = 3,
            doc_token_fill: float = -1e7,
            lr: float = 1e-4,
            out_act_fn: str = "relu",
            genre_act_fn: str = "sigmoid",
            pooler_type: str = "amax",
            n_genre: int = N_GENRES,
            glyph_type: str = "lexic",
            dropout: float = 0.1,
            norm_eps: float = 1e-6,
            n_strokes: int = 32,
            n_connect: int = 5,
            recon_weight: float = 1.,
            postit_weight: float = 1.,
            decode_weight: float = 1.,
            topology_weight: float = 1.,
            cosine_weight: float = 1.,
            genre_weight: float = .1,
            sparsity_weight: float = 0.05,
            sig_weight: float = 0.2,
            annealing: float = .001,
            n_eval_points: int = 32,
            consistent_hierarchy: bool = False,
            betas: Tuple[float, float] = (.9, 0.999),
            optim_params: Dict = {},
            scheduler_params: Dict | None = None,
            warmup_params: Dict | None = None,
            render_params: Dict | None = None,
            character_dict: Dict | None = None,
            **kwargs
    ):
        super(Inkler, self).__init__()
        if d_model < 1:
            raise ValueError("d_model must be larger than 0")
        self.d_model = d_model

        if seq_len < 1:
            raise ValueError("seq_len must be larger than 0")
        self.seq_len = seq_len

        if n_enc_blocks < 1:
            raise ValueError("n_enc_blocks must be larger than 0")
        self.n_enc_blocks = n_enc_blocks

        if n_dec_blocks < 1:
            raise ValueError("n_dec_blocks must be larger than 0")
        self.n_dec_blocks = n_dec_blocks

        if n_hierarchy < 1:
            raise ValueError("n_hierarchy must be larger than 0")
        self.n_hierarchy = n_hierarchy
        self.consistent_hierarchy = consistent_hierarchy
        self.n_strokes = n_strokes
        self.register_buffer("doc_token", torch.ones((1, 1, self.d_model)) * doc_token_fill)
        self.inkler_embedder = sdp.StructuralDocumentProcessor(
            d_model=self.d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_blocks=self.n_enc_blocks,
            dropout=dropout,
            n_hierarchy=self.n_hierarchy,
            n_xca=n_xca,
            pooler_type=pooler_type,
        )
        self.inkler_decoder = sdp.StructuralDocumentDecoder(
            d_model=self.d_model,
            n_heads=n_heads,
            seq_len=self.seq_len,
            d_ff=d_ff,
            n_blocks=self.n_dec_blocks,
            dropout=dropout,
            n_hierarchy=self.n_hierarchy,
            n_xca=n_xca,
        )

        if n_eval_points < 1:
            raise ValueError("n_eval_points must be larger than 0")
        self.n_eval_points = n_eval_points

        if render_params is None:
            render_params = {}

        self.n_connect = n_connect
        max_stroke = self.n_strokes if self.consistent_hierarchy else n_strokes * 2 ** (self.n_hierarchy + 1)
        self.glyph_decoder = GlyphDecoder(
            d_model=self.d_model,
            n_size=max_stroke * self.n_eval_points,
            n_params=2
        )
        self.inkler_genre = InklerGenre(
            n_genre=n_genre,
            genre_fn=genre_act_fn,
            d_model=d_model,
            act_fn=out_act_fn,
            norm_eps=norm_eps
        )
        self.renderer = GlyphRenderer(**render_params)

        if character_dict is None:
            character_dict = {}
        if glyph_type.lower() == "lexic":
            self.inkler_glyph = LexicInklerGlyph(d_model=self.d_model, n_strokes=max_stroke)
            self.character = LexicCharacterStrokes(**character_dict)
            self.contact_loss = lexic_contact_loss
        elif glyph_type.lower() == "bezier":
            self.inkler_glyph = BezierInklerGlyph(d_model=self.d_model, n_strokes=max_stroke)
            self.contact_loss = bezier_contact_loss
            self.character = BezierCharacterStrokes(**character_dict)
        elif glyph_type.lower() == "ode":
            self.inkler_glyph = ODEInklerGlyph(d_model=self.d_model, n_strokes=max_stroke)
            self.contact_loss = ode_contact_loss
            self.character = ODECharacterStrokes(**character_dict)
        elif glyph_type.lower() == "woodlouse":
            self.inkler_glyph = WoodlouseInklerGlyph(d_model=self.d_model, n_strokes=max_stroke)
            self.contact_loss = woodlouse_contact_loss
            self.character = WoodlouseCharacterStrokes(**character_dict)
        else:
            raise ValueError("Glyph type not supported. Select between lexic, bezier, and ode")

        self.lr = lr
        self.optim_dict = optim_params
        self.betas = betas
        self.scheduler_dict = scheduler_params
        self.warmup_dict = warmup_params
        self.recon_w = recon_weight
        self.sparsity_w = sparsity_weight
        self.genre_w = genre_weight
        self.postit_w = postit_weight
        self.decode_w = decode_weight
        self.topology_w = topology_weight
        self.cosine_w = cosine_weight
        self.sig = sig_weight
        self.annealing = annealing
        self.class_loss = torch.nn.BCELoss(reduction="mean")
        self.recon_loss = torch.nn.MSELoss(reduction="mean")
        self.dec_loss = torch.nn.MSELoss(reduction="mean")
        self.glyph_log_dir = None
        self.register_buffer("val_step", torch.tensor(-1.))
        self.register_buffer("train_step", torch.tensor(-1.))

    def configure_optimizers(self) -> OptimizerLRScheduler:
        logger.info(f"Use RAdam optimizer with lr {self.lr} and betas {self.betas}")
        optimizer = torch.optim.RAdam(
            self.parameters(),
            lr=self.lr,
            betas=self.betas,
            **self.optim_dict,
        )
        if self.warmup_dict is not None:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer=optimizer,
                **self.warmup_dict
            )
            logger.info("Warmup scheduler configured")
        else:
            warmup_scheduler = None
        if self.scheduler_dict is not None:
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                **self.scheduler_dict
            )
            logger.info("Cosine scheduler configured")
        else:
            cosine_scheduler = None
        if warmup_scheduler is not None and cosine_scheduler is not None:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer=optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_dict["total_iters"]],
            )
        elif cosine_scheduler is not None:
            scheduler = cosine_scheduler
        else:
            scheduler = None
        optim_dict = {"optimizer": optimizer}
        if scheduler is not None:
            optim_dict["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "step",
            }

        return optim_dict

    def forward(
            self,
            x_sentence: torch.Tensor,
            hierarchy: torch.Tensor,
            attention_mask: torch.Tensor,
    ) -> Dict:
        document_x, level_dict, deplex_idx, deplex_mask = self.inkler_embedder(
            x=x_sentence,
            hierarchy_x=self.doc_token.expand(x_sentence.shape[0], -1, -1),
            hierarchy=hierarchy,
            attention_mask=attention_mask,
        )

        inkler_glyph = {}
        glyph_decoder = {}
        stroke_points = {}

        inkler_glyph[f"glyph_{self.n_hierarchy - 1}"] = self.inkler_glyph(document_x)
        stroke_topology = get_stroke_topology(
            n_strokes=self.n_strokes if self.consistent_hierarchy else self.n_strokes * 2 ** (self.n_hierarchy + 1),
            n_connect=self.n_connect,
            use_connector_stroke=True,
        ).to(document_x.device)

        stroke_points[f"stroke_{self.n_hierarchy - 1}"] = self.character(
            inkler_glyph[f"glyph_{self.n_hierarchy - 1}"],
            topology=stroke_topology,
            add_noise=True
        )
        # perm_idx = torch.randperm(stroke_points[f"stroke_{self.n_hierarchy - 1}"].shape[2])
        glyph_decoder[f"recon_{self.n_hierarchy - 1}"] = self.glyph_decoder(
            stroke_points[f"stroke_{self.n_hierarchy - 1}"]#[:, :, perm_idx]
        )

        decode_x = self.inkler_decoder(
            x=glyph_decoder[f"recon_{self.n_hierarchy - 1}"],
            deplex_idx=deplex_idx,
            deplex_mask=deplex_mask,
            hierarchy_x=self.doc_token.expand(x_sentence.shape[0], -1, -1),
        )
        level_dict = {f"enc_{level}": v for level, v in level_dict.items()}
        decoder_dict = {f"dec_{level - 1}": v for level, v in enumerate(reversed(decode_x))}

        for level, embedding in decoder_dict.items():
            level = int(level.replace("dec_", ""))
            inkler_glyph[f"glyph_{level}"] = self.inkler_glyph(embedding)
            stroke_topology = get_stroke_topology(
                n_strokes=self.n_strokes if self.consistent_hierarchy else self.n_strokes * 2**(level + 2),
                n_connect=self.n_connect,
                use_connector_stroke=True,
            ).to(document_x.device)
            stroke_points[f"stroke_{level}"] = self.character(
                inkler_glyph[f"glyph_{level}"],
                topology=stroke_topology,
                add_noise=True
            )
            # perm_idx = torch.randperm(stroke_points[f"stroke_{level}"].shape[2])
            glyph_decoder[f"recon_{level}"] = self.glyph_decoder(stroke_points[f"stroke_{level}"])#[:, :, perm_idx])

        # score = self.inkler_score(document_x)
        genre = self.inkler_genre(document_x)
        deplex_mask = {f"deplex_mask_{k}": m for k, m in deplex_mask.items()}
        out_dict =  {
            "document_x": document_x,
            "genre": genre,
            "n_levels": len(level_dict),
            **level_dict,
            **decoder_dict,
            **inkler_glyph,
            **glyph_decoder,
            **stroke_points,
            **deplex_mask
        }

        return out_dict

    def loss(
            self,
            fwd_out: Dict,
            true_genre: torch.Tensor,
            postits: torch.Tensor | None = None,
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        genre_e = self.class_loss(fwd_out["genre"], true_genre)
        recon_e = torch.tensor(0., device=fwd_out["genre"].device)
        dec_e = torch.tensor(0., device=fwd_out["genre"].device)
        topology_e = torch.tensor(0., device=fwd_out["genre"].device)
        for level in range(-1, fwd_out["n_levels"] - 1):  # shifted by 1
            recon_e = recon_e + self.recon_loss(
                fwd_out[f"recon_{level}"][~fwd_out[f"deplex_mask_{level}"]],
                fwd_out[f"enc_{level}"][~fwd_out[f"deplex_mask_{level}"]].detach()
            )
            if level < fwd_out["n_levels"] - 2:  # one less decoding
                # TODO is this loss still needed with the document glyph bottleneck?
                dec_e = dec_e + self.dec_loss(
                    fwd_out[f"dec_{level}"][~fwd_out[f"deplex_mask_{level}"]],
                    fwd_out[f"enc_{level}"][~fwd_out[f"deplex_mask_{level}"]].detach()
                )
            stroke_topology = get_stroke_topology(
                n_strokes=self.n_strokes if self.consistent_hierarchy else self.n_strokes * 2**(level + 2),
                n_connect=self.n_connect,
                use_connector_stroke=True,
            ).to(fwd_out["genre"].device)
            topology_e = topology_e + self.contact_loss(
                fwd_out[f"stroke_{level}"],
                topology=stroke_topology,
            )
        norm_doc_x = fwd_out["document_x"] / torch.norm(fwd_out["document_x"], dim=-1, keepdim=True)
        batch_size = norm_doc_x.shape[0]
        cosine_sim = (norm_doc_x.squeeze(1) @ norm_doc_x.squeeze(1).transpose(-1, -2)).triu(1).sum() / torch.arange(
            batch_size,
            dtype=torch.float,
            device=self.device
        ).sum()
        cosine_sim = cosine_sim.abs()

        recon_e = recon_e / float(fwd_out["n_levels"])
        dec_e = dec_e / float(fwd_out["n_levels"] - 1)
        topology_e = topology_e / float(fwd_out["n_levels"])
        postit_e = torch.tensor(0., device=fwd_out["genre"].device)
        if postits is not None:
            # Equivalent to `self.mse(postits[not_inf_mask], document_x[not_inf_mask])`
            # guarded by `if torch.any(not_inf_mask)`, but avoids the boolean-mask
            # compaction (`nonzero`) and the `if tensor:` host sync, both of which
            # block the accelerator pipeline on every training step.
            not_inf_mask = torch.isfinite(postits)
            sq_err = torch.where(
                not_inf_mask,
                (postits.masked_fill(~not_inf_mask, 0.) - fwd_out["document_x"].masked_fill(~not_inf_mask, 0.)) ** 2,
                torch.zeros_like(postits),
            )
            count = not_inf_mask.sum().clamp(min=1).to(sq_err.dtype)
            postit_e = sq_err.sum() / count
        return {
            "genre_loss": genre_e,
            "postit_loss": postit_e,
            "recon_loss": recon_e,
            "decoder_loss": dec_e,
            "topology_loss": topology_e,
            "document_cosine_sim": cosine_sim,
            "total_loss": (
                    self.genre_w * genre_e
                    + self.postit_w * postit_e
                    + self.recon_w * recon_e
                    + self.decode_w * dec_e
                    + self.topology_w * topology_e
                    + self.cosine_w * cosine_sim
            )
        }

    @torch.no_grad()
    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
        self.train_step += 1

    def training_step(
            self,
            batch: Dict,
    ):
        result_dict = self(
            x_sentence=batch["texts"],
            hierarchy=batch["hierarchy"],
            attention_mask=batch["masks"],
        )
        utils.log_mps_memory("After forward", device=self.device.type)

        loss_dict = self.loss(
            fwd_out=result_dict,
            true_score=batch["scores"],
            true_genre=batch["genres"],
            postits=batch.get("postits", None),
        )
        utils.log_mps_memory("After loss calculation", device=self.device.type)

        if hasattr(self.inkler_glyph, "temperature"):
            self.inkler_glyph.temperature = torch.exp(-self.train_step * self.annealing)
        self.log(name="train_loss", value=loss_dict["total_loss"], on_step=True, on_epoch=False, prog_bar=True)
        for loss_name, loss_val in loss_dict.items():
            if loss_name != "total_loss":
                self.log(name=f"train_{loss_name}", value=loss_val, on_step=True, on_epoch=True, prog_bar=True)

        if self.device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
            utils.log_mps_memory("After emptying MPS cache", device=self.device.type)

        return loss_dict["total_loss"]

    @torch.no_grad()
    def validation_step(self, batch: Dict):
        result_dict = self(
            x_sentence=batch["texts"],
            hierarchy=batch["hierarchy"],
            attention_mask=batch["masks"],
        )
        loss = self.loss(
            fwd_out=result_dict,
            true_score=batch["scores"],
            true_genre=batch["genres"],
            postits=batch.get("postits", None),
        )
        for loss_name, loss_val in loss.items():
            self.log(name=f"val_{loss_name}", value=loss_val, on_step=False, on_epoch=True, prog_bar=True)

        if self.glyph_log_dir is None:
            self.glyph_log_dir = os.path.join("/".join(self.logger.log_dir.split("/")[:-2]), "val_glyphs")
            Path(self.glyph_log_dir).mkdir(parents=True, exist_ok=True)

        val_glyphs = {key: val for key, val in result_dict.items() if "glyph"
                      in key and int(key.split("_")[-1]) == self.n_hierarchy - 1}
        for glyph_level, glyph in val_glyphs.items():
            try:
                level = int(glyph_level.replace("glyph_", ""))
            except:
                break
            topology = get_stroke_topology(
                n_strokes=self.n_strokes if self.consistent_hierarchy else self.n_strokes * 2**(level + 2),
                n_connect=3 + level if self.consistent_hierarchy else 5,
                use_connector_stroke=True,
            )
            # rendered_glyph = (self.renderer(glyph, topology) > .5).float()
            pg = self.character(glyph, topology=topology, n_points=64)
            rendered_glyph = (self.renderer(pg) > .5).float()
            for i_glyph, rg in enumerate(rendered_glyph):
                save_name = f"{"_".join(batch["titles"][i_glyph].split(" "))}_BY_{
                    "_".join(batch["authors"][i_glyph].split(" "))}_epoch_{self.val_step}.png"
                fig, ax = plt.subplots(1, 1, figsize=(2, 2))
                ax.imshow(rg.detach().cpu().numpy(), aspect="auto", cmap="gray_r", interpolation="none")
                ax.axis("off")
                fig.tight_layout()
                fig.savefig(f"{self.glyph_log_dir}/{save_name}")
                plt.close(fig)
            del rendered_glyph
        plt.close("all")
        return loss["total_loss"]

    @torch.no_grad()
    def on_validation_epoch_start(self):
        self.val_step += 1

    @torch.no_grad()
    def test_step(self, batch: Dict):
        result_dict = self(
            x_sentence=batch["text"],
            document_ids=batch["document_ids"],
            attention_mask=batch["attention_mask"],
            postit=batch.get("postit", None)
        )
        loss = self.loss(
            fwd_out=result_dict,
            true_score=batch["score"],
            true_genre=batch["genre"],
        )
        for loss_name, loss_val in loss.items():
            self.log(name=f"test_{loss_name}", value=loss_val, on_step=False, on_epoch=True, prog_bar=False)

        return loss["total_loss"]
