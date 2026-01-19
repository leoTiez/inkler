import numpy as np
from typing import Tuple
import torch


def canonical_mask(mask: torch.Tensor, target_type: torch.dtype):
    if not torch.is_floating_point(mask):
        mask = torch.zeros_like(mask, dtype=target_type).masked_fill_(
            mask, float("-inf")
        )
    return mask.to(target_type)


class Pooler(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            pooling: str = "amax",
    ):
        super(Pooler, self).__init__()
        self.d_model = d_model
        if pooling.lower() not in ["mean", "amax", "prod", "amin"]:
            raise ValueError("pooing not understood.")

        self.pooling = pooling.lower()

    def forward(
            self,
            x: torch.Tensor,
            document_id: torch.Tensor,
            mask: torch.Tensor | None = None,
            token_size: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_ = x[:, token_size:]  # remove hierarchy
        batch_size, seq_len, x_dim = x_.shape
        if mask is None:
            mask = torch.zeros(x_.shape[:-1], device=x_.device, dtype=torch.bool)
        if mask.is_floating_point():
            mask = mask.to(torch.bool)
        mask = ~mask
        document_id = document_id.to(torch.int)
        min_val = torch.where(mask, document_id, torch.inf).min(-1)[0].to(torch.int)
        range_size = document_id.max(-1)[0] - min_val
        out_size = torch.max(range_size) + 1  # plus one to convert from index to size
        t_ = -torch.ones((batch_size * out_size, x_dim), device=x_.device)

        new_mask = torch.zeros((batch_size, out_size), dtype=torch.bool, device=x_.device)
        new_mask[torch.arange(out_size).reshape(1, -1) > range_size.reshape(-1, 1)] = True
        add_idx = -min_val.reshape(-1, 1) + torch.arange(batch_size).reshape(-1, 1) * (out_size - 1) + torch.arange(
            batch_size).reshape(-1, 1)

        t_.index_reduce_(
            0,
            (document_id + add_idx)[mask].flatten(),
            x_[mask].reshape(-1, x_dim),
            reduce=self.pooling,
            include_self=False
        )
        t_ = t_.reshape(batch_size, out_size, x_dim)
        return t_, new_mask


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
            proj_drop: float = 0.1
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        batch_size, seq_len, x_dim = x.shape
        mask = mask if mask is not None else torch.zeros_like(x[..., -1])
        mask = canonical_mask(mask, target_type=mask.dtype)
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, x_dim // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # size (q, k, v) x batch x heads x seq x model_d / n_heads
        q, k, v = qkv[0], qkv[1], qkv[2]

        # transpose and compute over features rather than seq
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
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


class AttentionWrapper(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = .1, batch_frist: bool = True):
        super(AttentionWrapper, self).__init__()
        self.multihead_attn = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=batch_frist,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        mask = mask if mask is not None else torch.zeros_like(x[..., -1])
        mask = canonical_mask(mask, target_type=mask.dtype)
        return self.multihead_attn(x, x, x, key_padding_mask=mask)[0]


class StructureTransformer(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ff: int,
            use_xca: bool,
            qkv_bias: bool = False,
            dropout: float = .1,
    ):
        super(StructureTransformer, self).__init__()
        if d_model < 1:
            raise ValueError("d_model must be a positive integer")
        self.d_model = d_model

        if n_heads < 1:
            raise ValueError("n_heads must be a positive integer")
        if self.d_model % n_heads != 0:
            raise ValueError("n_heads must be a divisor of d_model")
        self.n_heads = n_heads

        if d_ff < 1:
            raise ValueError("d_ff must be a positive integer")
        self.d_ff = d_ff

        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 <= dropout < 1")

        self.dropout = dropout
        self.skip1_norm = torch.nn.LayerNorm(self.d_model)
        self.skip2_norm = torch.nn.LayerNorm(self.d_model)
        if use_xca:
            self.structure_attention = XCA(
                d_model=self.d_model,
                n_heads=self.n_heads,
                qkv_bias=qkv_bias,
                proj_drop=self.dropout,
                attn_drop=self.dropout,
            )
        else:
            self.structure_attention = AttentionWrapper(
                d_model=self.d_model,
                n_heads=self.n_heads,
                dropout=self.dropout
            )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, self.d_ff),
            torch.nn.GELU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.d_ff, self.d_model),
            torch.nn.Dropout(self.dropout)
        )

    def forward(
            self,
            x: torch.Tensor,
            hierarchy_x: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            mask_hierarchy: bool = False
    ) -> torch.Tensor:
        batch_size, seq_len, x_dim = x.shape
        if attention_mask is not None:
            attention_mask = torch.hstack([
                torch.zeros(batch_size, hierarchy_x.shape[1], dtype=attention_mask.dtype, device=attention_mask.device),
                attention_mask
            ])
        else:
            attention_mask = torch.zeros(batch_size, seq_len + hierarchy_x.shape[1], dtype=torch.bool, device=x.device)

        if mask_hierarchy:
            attention_mask[:, :hierarchy_x.shape[1]] = torch.ones(
                hierarchy_x.shape[1],
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
        if torch.is_floating_point(attention_mask):
            attention_mask = attention_mask.to(torch.bool)

        x_ = self.structure_attention(
            torch.hstack([hierarchy_x, x]),
            mask=attention_mask,
        )

        # skip connection
        x_[:, hierarchy_x.shape[1]:, :] = x_[:, hierarchy_x.shape[1]:, :] + x
        x_ = self.skip1_norm(x_)
        x_ = self.mlp(x_)

        # skip connection
        x_[:, hierarchy_x.shape[1]:, :] = x_[:, hierarchy_x.shape[1]:, :] + x
        return self.skip2_norm(x_)


class SDPBlock(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ff: int,
            dropout: float = .1,
            n_xca: int = 2,
            n_hierarchy: int = 3,
            pooler_type: str = "token",
    ):
        super(SDPBlock, self).__init__()
        if n_xca > n_hierarchy:
            raise ValueError("n_xca must be less than or equal to n_hierarchy")
        struture_embedding = [StructureTransformer(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            use_xca=True
        ) for _ in range(n_xca)]
        st_layers = [
            StructureTransformer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_xca=False
            ) for _ in range(n_hierarchy - n_xca)
        ]
        self.st_layers = torch.nn.ModuleList([
            *struture_embedding,
            *st_layers
        ])

        self.pool = Pooler(d_model, pooling=pooler_type)

    def forward(
            self,
            x: torch.Tensor,
            hierarchy_x: torch.Tensor,
            hierarchy: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            mask_hierarchy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        sentence_embedding = None
        attention_mask_ = attention_mask.clone()
        if attention_mask.is_floating_point():
            attention_mask = attention_mask.to(torch.bool)
        attention_mask = ~attention_mask
        hid = hierarchy[..., -1]
        for i_d, st_layer in enumerate(self.st_layers):
            x = st_layer(x, hierarchy_x, attention_mask_, mask_hierarchy=mask_hierarchy)
            if i_d == 0:
                # remove hierarchy token
                sentence_embedding = x.clone()[:, 1:, :]
            x, attention_mask_ = self.pool(
                x,
                hid,
                mask=attention_mask_,
                token_size=hierarchy_x.shape[1]
            )
            if i_d < len(self.st_layers) - 1:
                hid = -torch.ones(np.prod(x.shape[:2]))
                min_val = torch.where(attention_mask, hierarchy[..., -(i_d + 1)], torch.inf).min(-1)[0].to(torch.int)
                add_idx = -min_val.reshape(-1, 1) + torch.arange(
                    x.shape[0]).reshape(-1, 1) * (x.shape[1] - 1) + torch.arange(x.shape[0]).reshape(-1, 1) # TODO ADD AS ABOVE
                hid = hid.index_reduce_(
                    0,
                    (hierarchy[..., -(i_d + 1)] + add_idx)[attention_mask].to(torch.int),
                    hierarchy[..., -(i_d + 2)][attention_mask],
                    reduce="amax",
                    include_self=False
                ).reshape(x.shape[:2])

        return sentence_embedding, x


class StructuralDocumentProcessor(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            seq_len: int,
            d_ff: int,
            n_blocks: int = 8,
            dropout: float = .1,
            n_xca: int = 2,
            n_hierarchy: int = 3,
            pooler_type: str = "token",
    ):
        super(StructuralDocumentProcessor, self).__init__()
        if d_model < 1:
            raise ValueError("d_model must be a positive integer")
        self.d_model = d_model

        if n_blocks < 1:
            raise ValueError("n_blocks must be a positive integer")
        self.n_blocks = n_blocks

        if seq_len < 1:
            raise ValueError("seq_len must be a positive integer")
        self.seq_len = seq_len
        self.register_buffer(
            "pe",
            self._create_pe(self.seq_len, self.d_model)
        )

        self.sdp_blocks = torch.nn.ModuleList([
                SDPBlock(
                    d_model=self.d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_hierarchy=n_hierarchy,
                    n_xca=n_xca,
                    pooler_type=pooler_type,
                ) for _ in range(self.n_blocks)
        ])

    @staticmethod
    def _create_pe(seq_len: int, d_model: int, scaling: float = 1e5):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(scaling / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(
            self,
            x: torch.Tensor,
            hierarchy_x: torch.Tensor,
            hierarchy: torch.Tensor,
            attention_mask: torch.Tensor | None = None
    ):
        batch_size, seq_len, x_dim = x.shape
        pe = self.pe[:seq_len].unsqueeze(0)
        # add lowest hierarchy to position
        x = x + pe + hierarchy[:, :, -1].unsqueeze(-1)
        for i_block, sdp_block in enumerate(self.sdp_blocks):
            x, hierarchy_x = sdp_block(x, hierarchy_x, hierarchy, attention_mask, mask_hierarchy=i_block == 0)

        return hierarchy_x

