from typing import Callable, Tuple, Dict

import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from transformers.activations import ACT2FN
import pytorch_lightning as pl

from inkler import sdp
from inkler.constants import N_GENRES


class InklerOut(torch.nn.Module):
    def __init__(self, d_model: int = int(2**9), act_fn: str | Callable | None = None, norm_eps: float = 1e-6):
        super(InklerOut, self).__init__()
        self.d_model = d_model
        self.flatten = torch.nn.Flatten()
        self.norm = torch.nn.LayerNorm(d_model, eps=norm_eps)
        self.dense = torch.nn.Linear(self.d_model, self.d_model)
        if act_fn is None:
            self.act_fn = lambda _: _
        elif isinstance(act_fn, str):
            self.act_fn = ACT2FN[act_fn]
        else:
            self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.dense(self.norm(x))
        return self.act_fn(x)


class InklerScore(InklerOut):
    def __init__(self, d_model: int = int(2**9), act_fn: str | Callable | None = torch.sigmoid, norm_eps: float = 1e-6):
        super(InklerScore, self).__init__(d_model=d_model, act_fn=act_fn, norm_eps=norm_eps)
        self.out = torch.nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super(InklerScore, self).forward(x)
        return self.out(x)


class InklerGenre(InklerOut):
    def __init__(
            self,
            n_genre: int = N_GENRES,
            d_model: int = int(2**9),
            act_fn: str | Callable | None = torch.sigmoid,
            norm_eps: float = 1e-6
    ):
        super(InklerGenre, self).__init__(d_model=d_model, act_fn=act_fn, norm_eps=norm_eps)
        self.out = torch.nn.Linear(d_model, n_genre)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super(InklerGenre, self).forward(x)
        return torch.sigmoid(self.out(x))


class Inkler(pl.LightningModule):
    def __init__(
            self,
            d_model: int = int(2**9),
            seq_len: int = int(2**15),
            n_heads: int = 16,
            d_ff: int = int(2**10),
            n_blocks: int = 8,
            n_xca: int = 2,
            n_hierarchy: int = 3,
            doc_token_fill: float = -1e7,
            lr: float = 1e-4,
            score_act_fn: str = "relu",
            genre_act_fn: str = "relu",
            pooler_type: str = "amax",
            n_genre: int = N_GENRES,
            dropout: float = 0.1,
            norm_eps: float = 1e-6,
            postit_weight: float = 1.,
            score_weight: float = 1.,
            genre_weight: float = 1.,
            betas: Tuple[float, float] = (.9, 0.999),
            huber_beta: float = .01,
            optim_params: Dict = {},
            scheduler_params: Dict = {},
            **kwargs
    ):
        super(Inkler, self).__init__()
        if d_model < 1:
            raise ValueError("d_model must be larger than 0")
        self.d_model = d_model

        if seq_len < 1:
            raise ValueError("seq_len must be larger than 0")
        self.seq_len = seq_len

        if n_blocks < 1:
            raise ValueError("n_blocks must be larger than 0")
        self.n_blocks = n_blocks

        if n_hierarchy < 1:
            raise ValueError("n_hierarchy must be larger than 0")
        self.n_hierarchy = n_hierarchy
        self.register_buffer("doc_token", torch.ones((1, 1, self.d_model)) * doc_token_fill)
        self.inkler_embedder = sdp.StructuralDocumentProcessor(
            d_model=self.d_model,
            n_heads=n_heads,
            seq_len=self.seq_len,
            d_ff=d_ff,
            n_blocks=self.n_blocks,
            dropout=dropout,
            n_hierarchy=self.n_hierarchy,
            n_xca=n_xca,
            pooler_type=pooler_type,
        )

        self.inkler_score = InklerScore(
            d_model=d_model,
            act_fn=score_act_fn,
            norm_eps=norm_eps
        )
        self.inkler_genre = InklerGenre(
            n_genre=n_genre,
            d_model=d_model,
            act_fn=genre_act_fn,
            norm_eps=norm_eps
        )

        self.lr = lr
        self.optim_dict = optim_params
        self.betas = betas
        self.scheduler_dict = scheduler_params
        self.score_w = score_weight
        self.genre_w = genre_weight
        self.postit_w = postit_weight
        self.class_loss = torch.nn.BCELoss(reduction="mean")
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_beta)
        self.mse = torch.nn.MSELoss()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.RAdam(
            self.parameters(),
            lr=self.lr,
            betas=self.betas,
            **self.optim_dict,
        )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer=optimizer,
        #     **self.scheduler_dict
        # )
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "interval": "step",
            # }
        }

    def forward(
            self,
            x_sentence: torch.Tensor,
            hierarchy: torch.Tensor,
            attention_mask: torch.Tensor,
    ) -> Dict:
        document_x = self.inkler_embedder(
            x=x_sentence,
            hierarchy_x=self.doc_token.tile(dims=(x_sentence.shape[0], 1, 1)),
            hierarchy=hierarchy,
            attention_mask=attention_mask,
        )

        score = self.inkler_score(document_x)
        genre = self.inkler_genre(document_x)

        out_dict =  {
            "score": score,
            "genre": genre,
            "document_x": document_x,
        }

        return out_dict

    def loss(
            self,
            fwd_out: Dict,
            true_score: float | torch.Tensor,
            true_genre: torch.Tensor,
            postits: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        score_e = self.huber_loss(fwd_out["score"], true_score.reshape(fwd_out["score"].shape))
        genre_e = self.class_loss(fwd_out["genre"], true_genre)
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
            "score_loss": score_e,
            "genre_loss": genre_e,
            "postit_loss": postit_e,
            "total_loss": self.score_w * score_e + self.genre_w * genre_e + self.postit_w * postit_e
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

        return loss["total_loss"]

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
