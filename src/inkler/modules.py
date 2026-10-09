from typing import Tuple
import torch
from torch.nn import functional as F

from inkler.utils import index_reduce_mps, canonical_mask, child_order_within_parent
from inkler.constants import BUCKETS


class LPI(torch.nn.Module):
    """
    local patch interaction
    """
    def __init__(self, d_model, ks=5):
        super().__init__()
        self.d_model = d_model
        self.ks = ks
        self.conv = torch.nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_size=self.ks,
            padding=self.ks // 2,
            groups=self.d_model  # independent convolution per channels
        )
        self.act = torch.nn.GELU()

    def forward(self, x, mask):
        x = self.conv(x.masked_fill(mask.unsqueeze(-1), 0.).transpose(-1, -2))
        return self.act(x).transpose(-1, -2)


class Pooler(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            pooling: str = "amax",
    ):
        super(Pooler, self).__init__()
        self.d_model = d_model
        if pooling.lower() not in ["mean", "amax", "prod", "amin", "logmeanexp"]:
            raise ValueError("pooling not understood.")

        self.pooling = pooling.lower()
        self.temperature = torch.nn.Parameter(torch.randn((1, 1, self.d_model)))

    def forward(
            self,
            x: torch.Tensor,
            document_id: torch.Tensor,
            mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Pool results over indices"""
        # do not include cls token for pooling
        x_ = x # [:, token_size:]  # remove hierarchy
        batch_size, seq_len, x_dim = x_.shape
        if mask is None:
            mask = torch.zeros(x_.shape[:-1], device=x_.device, dtype=torch.bool)
        if mask.is_floating_point():
            mask = mask.to(torch.bool)
        mask = ~mask
        document_id = document_id.to(torch.int)
        # select how far indices range per data point in batch
        min_val = torch.where(mask, document_id, torch.inf).min(-1)[0].to(torch.int)
        max_val = torch.where(mask, document_id, -torch.inf).max(-1)[0].to(torch.int)
        range_size = max_val - min_val
        # create new output tensor that is as big as max range
        out_size = int(torch.max(range_size) + 1)  # plus one to convert from index to size
        dist_buckets = torch.tensor(BUCKETS, dtype=torch.float32) - out_size
        dist_buckets[dist_buckets < 0] = torch.inf
        # get first bucket that is larger than the max text length
        out_size = BUCKETS[torch.abs(dist_buckets).argmin()]
        t_ = -torch.ones((batch_size * out_size, x_dim), device=x_.device)

        # set mask such that values that don't reach out_size are masked out
        # (equivalent to filling a zeros tensor via a boolean-mask assignment, but
        # avoids the `nonzero` this triggers via boolean indexing on MPS/CUDA)
        new_mask = torch.arange(out_size, device=x_.device).reshape(1, -1) > range_size.reshape(-1, 1)
        # calculate offset indices to collate data values
        # offset is -min_val.reshape(-1, 1) for resetting to zero + size of new collated output tensor
        add_idx = -min_val.reshape(-1, 1) + torch.arange(
            batch_size,
            device=x_.device
        ).reshape(-1, 1) * out_size
        collate_idx = torch.where(mask, document_id + add_idx, -1)

        if self.pooling == "logmeanexp":
            x_ = torch.exp(x_ / F.softplus(self.temperature).expand(x_.shape).clip(min=0.05, max=10.))
            pooling = "mean"
        else:
            pooling = self.pooling
        # collate data values over indices that are aggregated over hierarchical document ids + offset,
        # index_reduce_mps is an mps-friendly implementation of the index_reduce tensor function
        t_, out_idx = index_reduce_mps(
            input=t_,
            dim=0,
            index=collate_idx.flatten(),
            source=x_.reshape(-1, x_dim),
            reduce=pooling,
            include_self=False,
            mask=mask.flatten(),
        )
        # reshape
        t_ = t_.reshape(batch_size, out_size, x_dim)
        if self.pooling == "logmeanexp":
            log_mask = t_ > 1e-8
            t_ = torch.where(
                log_mask,
                t_.clip(min=1e-8).log() * F.softplus(self.temperature).clip(min=0.05, max=10.),
                torch.zeros_like(t_)
            )

        # avoid gradient flow from out_idx into reconstruction
        return t_, new_mask.detach(), out_idx.detach() if out_idx is not None else None


class Unpooler(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            init_shift: float = -10.,
            max_children: int = 50000,
            dropout: float = 0.25,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_children = max_children
        self.token = torch.nn.Parameter(torch.randn((1, self.d_model)) + init_shift)
        self.pos_emb = torch.nn.Embedding(max_children, d_model)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(
            self,
            pooled: torch.Tensor,
            seq_len: int,
            hierarchy_idx: torch.Tensor,
            mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, out_size, x_dim = pooled.shape
        hierarchy_idx = hierarchy_idx.reshape(batch_size, seq_len)
        hierarchy_idx = hierarchy_idx.to(torch.int64)
        child_order = child_order_within_parent(hierarchy_idx)
        child_order = child_order.to(torch.int64)
        valid = (hierarchy_idx >= 0) & ~mask

        # route invalid entries to index 0 instead of boolean indexing (MPS-friendly)
        zeros = torch.zeros_like(hierarchy_idx)
        safe_parent = torch.where(valid, hierarchy_idx, zeros)
        safe_rank = torch.where(valid, child_order, zeros).clamp(max=self.pos_emb.num_embeddings - 1)

        # broadcast each parent vector to all of its children, then add structural position
        out = pooled.reshape(-1, x_dim)[
              safe_parent.flatten()
          ].reshape(batch_size, seq_len, x_dim) + self.dropout(self.pos_emb(safe_rank))

        # padding / invalid positions get the learned token
        return torch.where(valid.unsqueeze(-1), out, self.token.to(out.dtype))


class XCA(torch.nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)

    taken from the xcit paper
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.1,
            proj_drop: float = 0.1,
            ks: int = 7,
    ):
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be a positive integer")
        self.d_model = d_model

        if n_heads < 1:
            raise ValueError("n_heads must be a positive integer")
        self.n_heads = n_heads
        self.temperature = torch.nn.Parameter(torch.ones(n_heads, 1, 1))

        self.qkv = torch.nn.Linear(self.d_model, self.d_model * 3, bias=qkv_bias)  # times 3 for Q, K, and V
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(self.d_model, self.d_model)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.lpi = LPI(self.d_model, ks=ks)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        batch_size, seq_len, x_dim = x.shape
        mask = mask if mask is not None else torch.zeros_like(x[..., -1])
        mask = canonical_mask(mask, target_type=mask.dtype)
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, x_dim // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # size (q, k, v) x batch x heads x seq x model_d / n_heads
        q, k, v = qkv[0], qkv[1], qkv[2]

        # transpose and compute over features rather than seq
        # apply mask over sequence dimension
        q.masked_fill_(mask.unsqueeze(-1).unsqueeze(1), 0.)
        k.masked_fill_(mask.unsqueeze(-1).unsqueeze(1), 0.)
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(batch_size, seq_len, x_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.lpi(x, mask)  # local patch interaction
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


class AttentionWrapper(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = .1):
        super(AttentionWrapper, self).__init__()
        self.multihead_attn = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        mask = mask if mask is not None else torch.zeros_like(x[..., -1])
        mask = canonical_mask(mask, target_type=mask.dtype)
        return self.multihead_attn(x, x, x, key_padding_mask=mask)[0]