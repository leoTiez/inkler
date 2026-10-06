from typing import Callable

import torch
from transformers.activations import ACT2FN

from inkler.constants import N_GENRES


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


class GlyphDecoder(torch.nn.Module):
    def __init__(self, n_size=32, n_params: int = 9, d_model=384, act_thresh: float = .5):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_size * n_params, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, d_model)
        )
        self.act_thresh = act_thresh

    def forward(self, theta):
        return self.net(theta.flatten(1))