import numpy as np
from typing import Tuple
import torch

from inkler.modules import XCA, AttentionWrapper, Pooler
from inkler.utils import index_reduce_mps


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
        # use xca or conventional attention depending on how large sequence is.
        # sentences or paragraphs benefit from xca implementation
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
            # add class token mask
            attention_mask = torch.hstack([
                torch.zeros(batch_size, hierarchy_x.shape[1], dtype=attention_mask.dtype, device=attention_mask.device),
                attention_mask
            ])
        else:
            attention_mask = torch.zeros(batch_size, seq_len + hierarchy_x.shape[1], dtype=torch.bool, device=x.device)

        # mask hierarchy token if set. for first layer
        # useful
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

        # skip connection without class token
        x_[:, hierarchy_x.shape[1]:, :] = x_[:, hierarchy_x.shape[1]:, :] + x
        x_ = self.skip1_norm(x_)
        x_ = self.mlp(x_)

        # skip connection without class token
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
            pooler_type: str = "amax",
    ):
        super(SDPBlock, self).__init__()
        if n_xca > n_hierarchy:
            raise ValueError("n_xca must be less than or equal to n_hierarchy")
        # first xca based transformer
        struture_embedding = [StructureTransformer(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            use_xca=True
        ) for _ in range(n_xca)]
        # then attention based transformer when sequence length has been reduced
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
        if attention_mask.is_floating_point():
            attention_mask = attention_mask.to(torch.bool)
        # two attention masks, one that remains in passed dimensionality
        # the other shrinks with structural processing depth when sentences, paragraphs and chapters are collated
        attention_mask_ = attention_mask.clone()
        attention_mask = ~attention_mask
        # fetch lowest hierarchy which are paragraphs
        hid = hierarchy[..., -1]
        for i_d, st_layer in enumerate(self.st_layers):
            # hierarchy attention
            x = st_layer(x, hierarchy_x, attention_mask_, mask_hierarchy=mask_hierarchy)
            if i_d == 0:
                # fetch sentence embedding that is used as input for next layer
                # when i_d > 0, x continues being aggregated over hierarchy ids until class token is found
                sentence_embedding = x.clone()[:, 1:, :]  # remove hierarchy token

            # pool embedding over hierarchy ides
            x, attention_mask_ = self.pool(
                x,
                hid,
                mask=attention_mask_,
                token_size=hierarchy_x.shape[1]
            )
            # adapt hierarchy ids to new layout
            if i_d < len(self.st_layers) - 1:
                # compute offset and collation as for the pooling but mus be performed over unchanged attention masl
                hid = -torch.ones(np.prod(x.shape[:2]), device=x.device)
                min_val = torch.where(attention_mask, hierarchy[..., -(i_d + 1)], torch.inf).min(-1)[0].to(torch.int)
                add_idx = -min_val.reshape(-1, 1) + torch.arange(
                    x.shape[0],
                    device=x.device,
                ).reshape(-1, 1) * x.shape[1]
                # collate indices
                hid = index_reduce_mps(
                    input=hid,
                    dim=0,
                    index=(hierarchy[..., -(i_d + 1)] + add_idx)[attention_mask].to(torch.int),
                    source=hierarchy[..., -(i_d + 2)][attention_mask],
                    reduce="amax",
                    include_self=False
                ).reshape(x.shape[:2])

        # return sentence embedding and class token
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
            pooler_type: str = "amax",
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
            # x is new sentence embedding, hierarchy_x is new class token
            x, hierarchy_x = sdp_block(x, hierarchy_x, hierarchy, attention_mask, mask_hierarchy=i_block == 0)

        return hierarchy_x

