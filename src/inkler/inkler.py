import logging
from typing import Callable, Tuple, Dict
from pathlib import Path
import os
import math

import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from transformers.activations import ACT2FN
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from inkler import sdp
from inkler.constants import N_GENRES
from inkler.renderer import DifferentiableGlyphRenderer

logger = logging.getLogger(__name__)


class InklerOut(torch.nn.Module):
    def __init__(
            self,
            d_model: int = int(2**9),
            act_fn: str | Callable | None = "relu",
            norm_eps: float = 1e-6
    ):
        super(InklerOut, self).__init__()
        self.d_model = d_model
        self.flatten = torch.nn.Flatten()
        self.norm = torch.nn.LayerNorm(d_model, eps=norm_eps)
        self.dense = torch.nn.Linear(self.d_model, self.d_model)
        self.act_fn = self._convert_activation(act_fn)

    @staticmethod
    def _convert_activation(act_fn: str | Callable | None = None) -> Callable:
        if act_fn is None:
            return lambda _: _
        elif isinstance(act_fn, str):
            return ACT2FN[act_fn]
        else:
            return act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.dense(self.norm(x))
        return self.act_fn(x)


class InklerScore(InklerOut):
    def __init__(
            self,
            d_model: int = int(2**9),
            act_fn: str | Callable | None = "relu",
            score_fn: str | Callable | None = "linear",
            norm_eps: float = 1e-6
    ):
        super(InklerScore, self).__init__(d_model=d_model, act_fn=act_fn, norm_eps=norm_eps)
        self.out = torch.nn.Linear(d_model, 1)
        self.score_fn = self._convert_activation(score_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super(InklerScore, self).forward(x)
        return self.score_fn(self.out(x))


class InklerGenre(InklerOut):
    def __init__(
            self,
            n_genre: int = N_GENRES,
            d_model: int = int(2**9),
            act_fn: str | Callable | None = "relu",
            genre_fn: str | Callable | None = "sigmoid",
            norm_eps: float = 1e-6
    ):
        super(InklerGenre, self).__init__(d_model=d_model, act_fn=act_fn, norm_eps=norm_eps)
        self.out = torch.nn.Linear(d_model, n_genre)
        self.genre_fn = self._convert_activation(genre_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super(InklerGenre, self).forward(x)
        return self.genre_fn(self.out(x))


class InklerGlyph(torch.nn.Module):
    def __init__(self, d_model=384, n_strokes=16):
        super().__init__()
        self.n_strokes = n_strokes
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_model, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, n_strokes * 6)
        )

    def forward(self, z):
        theta = self.net(z).view(-1, self.n_strokes, 6)
        # Position in [0,1]
        xy = torch.sigmoid(theta[..., 0:2])
        # Length
        length = torch.sigmoid(theta[..., 2:3])
        # Angle
        angle = torch.tanh(theta[..., 3:4]) * torch.pi
        # Width
        width = torch.tanh(theta[..., 4:5]).relu()  # get only positive width
        # Curvature
        curvature = torch.tanh(theta[..., 5:6])

        return torch.cat([
            xy,
            length,
            angle,
            width,
            curvature
        ], dim=-1)


class GlyphDecoder(torch.nn.Module):
    def __init__(self, n_strokes=16, d_model=384):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_strokes * 6, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, d_model)
        )

    def forward(self, theta):
        return self.net(theta.flatten(1))


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
            dropout: float = 0.1,
            norm_eps: float = 1e-6,
            n_strokes: int = 32,
            recon_weight: float = 1.,
            postit_weight: float = 1.,
            decode_weight: float = 1.,
            genre_weight: float = .1,
            sparsity_weight: float = 0.05,
            sig_weight: float = 0.1,
            betas: Tuple[float, float] = (.9, 0.999),
            optim_params: Dict = {},
            scheduler_params: Dict | None = None,
            warmup_params: Dict | None = None,
            render_params: Dict | None = None,
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
        self.register_buffer("doc_token", torch.ones((1, 1, self.d_model)) * doc_token_fill)
        self.inkler_embedder = sdp.StructuralDocumentProcessor(
            d_model=self.d_model,
            n_heads=n_heads,
            seq_len=self.seq_len,
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

        self.inkler_glyph = InklerGlyph(d_model=d_model, n_strokes=n_strokes)
        self.glyph_decoder = GlyphDecoder(d_model=d_model, n_strokes=n_strokes)

        self.inkler_genre = InklerGenre(
            n_genre=n_genre,
            genre_fn=genre_act_fn,
            d_model=d_model,
            act_fn=out_act_fn,
            norm_eps=norm_eps
        )

        if render_params is None:
            render_params = {}
        self.renderer = DifferentiableGlyphRenderer(**render_params)

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
        self.sig = sig_weight
        self.class_loss = torch.nn.BCELoss(reduction="mean")
        self.recon_loss = torch.nn.MSELoss(reduction="mean")
        self.dec_loss = torch.nn.MSELoss(reduction="mean")
        self.glyph_log_dir = None
        self.val_step = -1

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
            hierarchy_x=self.doc_token.tile(dims=(x_sentence.shape[0], 1, 1)),
            hierarchy=hierarchy,
            attention_mask=attention_mask,
        )

        decode_x = self.inkler_decoder(
            x=level_dict,
            deplex_idx=deplex_idx,
            deplex_mask=deplex_mask,
            hierarchy_x=document_x,
        )

        inkler_glyph = {}
        glyph_decoder = {}
        for level, embedding in level_dict.items():
            inkler_glyph[f"glyph_{level}"] = self.inkler_glyph(
                embedding + self.sig * torch.randn_like(embedding)  # reparameterize
            )
            glyph_decoder[f"recon_{level}"] = self.glyph_decoder(inkler_glyph[f"glyph_{level}"])

        # score = self.inkler_score(document_x)
        genre = self.inkler_genre(document_x)

        level_dict = {f"enc_{level}": v for level, v in level_dict.items()}
        decoder_dict = {f"dec_{level - 1}": v for level, v in enumerate(reversed(decode_x))}
        out_dict =  {
            "document_x": document_x,
            "genre": genre,
            "n_levels": len(level_dict),
            **level_dict,
            **decoder_dict,
            **inkler_glyph,
            **glyph_decoder,
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
        sparsity_e = torch.tensor(0., device=fwd_out["genre"].device)
        for level in range(-1, fwd_out["n_levels"] - 1):  # shifted by 1
            recon_e = recon_e + self.recon_loss(
                fwd_out[f"recon_{level}"].reshape(-1, self.d_model), fwd_out[f"enc_{level}"].reshape(-1, self.d_model)
            )
            if level < fwd_out["n_levels"] - 2:  # one less decoding
                dec_e = dec_e + self.dec_loss(
                    fwd_out[f"dec_{level}"].reshape(-1, self.d_model), fwd_out[f"enc_{level}"].reshape(-1, self.d_model)
                )
            sparsity_e = sparsity_e + fwd_out[f"glyph_{level}"][..., 4].mean()
            sparsity_e = sparsity_e.mean()

        recon_e = recon_e / float(fwd_out["n_levels"])
        sparsity_e = sparsity_e / float(fwd_out["n_levels"])
        dec_e = dec_e / float(fwd_out["n_levels"] - 1)
        postit_e = torch.tensor(0.)
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
            "sparsity_loss": sparsity_e,
            "decoder_loss": dec_e,
            "total_loss": (
                    self.genre_w * genre_e
                    + self.postit_w * postit_e
                    + self.recon_w * recon_e
                    + self.sparsity_w * sparsity_e
                    + self.decode_w * dec_e
            )
        }

    def training_step(
            self,
            batch: Dict,
    ):
        result_dict = self(
            x_sentence=batch["texts"],
            hierarchy=batch["hierarchy"],
            attention_mask=batch["masks"],
        )
        loss_dict = self.loss(
            fwd_out=result_dict,
            true_score=batch["scores"],
            true_genre=batch["genres"],
            postits=batch.get("postits", None),
        )
        with torch.no_grad():
            self.log(name="train_loss", value=loss_dict["total_loss"], on_step=True, on_epoch=False, prog_bar=True)
            for loss_name, loss_val in loss_dict.items():
                if loss_name != "total_loss":
                    self.log(name=f"train_{loss_name}", value=loss_val, on_step=True, on_epoch=True, prog_bar=True)

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
            rendered_glyph = self.renderer(glyph)
            for i_glyph, rg in enumerate(rendered_glyph):
                save_name = f"{"_".join(batch["titles"][i_glyph].split(" "))}_BY_{
                    "_".join(batch["authors"][i_glyph].split(" "))}_epoch_{self.val_step}.png"
                fig, ax = plt.subplots(1, 1, figsize=(2, 2))
                ax.imshow(rg.detach().cpu().numpy(), aspect="auto", cmap="gray_r", interpolation="none")
                ax.axis("off")
                fig.tight_layout()
                fig.savefig(f"{self.glyph_log_dir}/{save_name}")
                plt.close(fig)
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
