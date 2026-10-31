import numpy as np
from typing import Tuple, Dict, List
import torch

from inkler.modules import XCA, AttentionWrapper, Pooler, Unpooler, EMA
from inkler.utils import index_reduce_mps, child_order_within_parent
from inkler.constants import BUCKETS


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
        x_ = x_[:, hierarchy_x.shape[1]:, :] + x
        return self.skip2_norm(x_)


class SDPEncoderBlock(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ff: int,
            dropout: float = .1,
            n_xca: int = 2,
            n_hierarchy: int = 3,
            pooler_type: str = "amax",
            var_min: float = 1e-4,
            ema_weight: float = .95,
    ):
        super().__init__()
        if n_xca > n_hierarchy:
            raise ValueError("n_xca must be less than or equal to n_hierarchy")
        # first xca based transformer
        self.d_model = d_model
        self.n_hierarchy = n_hierarchy
        struture_embedding = [StructureTransformer(
            d_model=self.d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            use_xca=True
        ) for _ in range(n_xca)]
        # then attention based transformer when sequence length has been reduced
        st_layers = [
            StructureTransformer(
                d_model=self.d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_xca=False
            ) for _ in range(self.n_hierarchy - n_xca)
        ]
        self.st_layers = torch.nn.ModuleList([
            *struture_embedding,
            *st_layers
        ])

        self.ema_forward = torch.nn.ModuleList([EMA(
            self.d_model,
            var_min=var_min,
            ema_weight=ema_weight
        ) for _ in range(self.n_hierarchy + 1)])

        self.ema_backward = torch.nn.ModuleList([EMA(
            self.d_model,
            var_min=var_min,
            ema_weight=ema_weight
        ) for _ in range(self.n_hierarchy + 1)])

        self.register_buffer(
            "pe",
            self._create_pe(max(BUCKETS), self.d_model)
        )
        self.gamma = torch.nn.Parameter(.1 * torch.ones(self.n_hierarchy, dtype=torch.float))
        self.pool = Pooler(d_model, pooling=pooler_type)

    @staticmethod
    def _create_pe(
            seq_len: int,
            d_model: int,
            scaling: float = 1e4,
            position: torch.Tensor | None = None,
            device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:

        if position is None:
            pe = torch.zeros((seq_len, d_model)).to(device).unsqueeze(0)
            position = torch.arange(0, seq_len, dtype=torch.float).to(device)
        else:
            pe = torch.zeros((position.shape[0], seq_len, d_model)).to(device)
        position = position.unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, d_model, 2).to(device).float() * (-np.log(scaling) / d_model))
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term)
        return pe

    def _embed_position(self, x: torch.Tensor, hierarchy: torch.Tensor | None) -> torch.Tensor:
        batch_size, seq_len, x_dim = x.shape
        if hierarchy is None:
            pe = self.pe[:seq_len].unsqueeze(0).float()
        else:
            with torch.no_grad():
                pe = torch.zeros_like(x)
                if hierarchy.dim() < 3:
                    hierarchy = hierarchy.unsqueeze(-1)
                for hid in range(hierarchy.shape[-1]):
                    child_order = child_order_within_parent(hierarchy[..., -hid]).float().to(x.device)
                    pe = pe + self.gamma[-hid] * self._create_pe(seq_len, x_dim, position=child_order, device=x.device)
        return x + pe

    def forward(
            self,
            x: torch.Tensor,
            hierarchy_x: torch.Tensor,
            hierarchy: torch.Tensor,
            attention_mask: torch.Tensor,
            mask_hierarchy: bool = False,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], List[torch.Tensor], Dict[int, torch.Tensor], Dict[str, torch.Tensor]]:
        out_embedding = {-1: self.ema_forward[0](x, attention_mask)}
        out_agg = {"agg_-1": self.ema_backward[0](x, attention_mask)}  # untrained weak reconstruction target
        deplex_idx, deplex_mask = [], {}
        hid_idx = []
        if attention_mask.is_floating_point():
            attention_mask = attention_mask.to(torch.bool)
        # two attention masks, one that remains in passed dimensionality
        # the other shrinks with structural processing depth when sentences, paragraphs and chapters are collated
        attention_mask_ = attention_mask.clone()
        deplex_mask[-1] = attention_mask_.clone()
        attention_mask = ~attention_mask
        # fetch lowest hierarchy which are paragraphs
        # hierarchy attention
        x = self._embed_position(x, hierarchy)
        hid = hierarchy[..., -1]
        for i_d, st_layer in enumerate(self.st_layers):
            x = st_layer(x, hierarchy_x, attention_mask_, mask_hierarchy=mask_hierarchy)

            # pool embedding over hierarchy ides
            x, attention_mask_, demu_idx = self.pool(
                x,
                hid,
                mask=attention_mask_,
            )
            # x = self.ema_forward[i_d + 1](x, mask=attention_mask_)
            deplex_idx.append(demu_idx.clone())
            deplex_mask[i_d] = attention_mask_.clone()
            out_embedding[i_d] = x.clone()

            # aggregate for unbiased centered target
            agg_ = -torch.ones_like(x).reshape(-1, self.d_model)
            min_val = torch.where(~deplex_mask[i_d - 1], hid, torch.inf).min(-1)[0].to(torch.int)
            add_idx = -min_val.reshape(-1, 1) + torch.arange(
                x.shape[0],
                device=x.device,
            ).reshape(-1, 1) * x.shape[1]
            collate_idx = torch.where(~deplex_mask[i_d - 1], hid + add_idx, -1).to(torch.int)
            agg_, _ = index_reduce_mps(
                input=agg_,
                dim=0,
                index=collate_idx.flatten(),
                source=out_agg[f"agg_{i_d - 1}"].reshape(-1, self.d_model).detach(),
                reduce="mean",
                include_self=False,
                mask=~deplex_mask[i_d - 1].flatten(),
                return_idx=False
            )
            agg = agg_.reshape(x.shape)
            agg = self.ema_backward[i_d + 1](agg, attention_mask_)
            out_agg[f"agg_{i_d}"] = agg.detach()

            if i_d < len(self.st_layers) - 1:
                # compute offset and collation as for the pooling but mus be performed over unchanged attention masl
                hid_ = -torch.ones(np.prod(x.shape[:2]), device=x.device)
                min_val = torch.where(attention_mask, hierarchy[..., -(i_d + 1)], torch.inf).min(-1)[0].to(torch.int)
                add_idx = -min_val.reshape(-1, 1) + torch.arange(
                    x.shape[0],
                    device=x.device,
                ).reshape(-1, 1) * x.shape[1]
                collate_idx = torch.where(attention_mask, hierarchy[..., -(i_d + 1)] + add_idx, -1).to(torch.int)
                # collate indices
                hid_, _ = index_reduce_mps(
                    input=hid_,
                    dim=0,
                    index=collate_idx.flatten(),
                    source=hierarchy[..., -(i_d + 2)].flatten(),
                    reduce="amax",
                    include_self=False,
                    mask=attention_mask.flatten(),
                    return_idx=False
                )
                hid = hid_.reshape(x.shape[:2])
                hid_idx.append(hid.clone())

        # return sentence embedding and class token
        return x, out_embedding, deplex_idx, deplex_mask, out_agg


class SDPDecoderBlock(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ff: int,
            dropout: float = .1,
            n_xca: int = 2,
            n_hierarchy: int = 3,
            skip_dropout: float = .8
    ):
        super().__init__()
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

        self.unpool = Unpooler(d_model)
        self.skip_norm_pre = torch.nn.LayerNorm(d_model)
        self.norm_x = torch.nn.LayerNorm(d_model)
        self.skip_dropout = skip_dropout

    def forward(
            self,
            x: torch.Tensor,
            deplex_idx: List[torch.Tensor],
            deplex_mask: List[torch.Tensor],
            encode_x: Dict[int, torch.Tensor],
            hierarchy_x: torch.Tensor,
            mask_hierarchy: bool = False
    ):
        batch_size = x.shape[0]
        decode_x = []
        for i_d in range(len(self.st_layers) - 1, -1, -1):
            st_layer = self.st_layers[i_d]
            demulidx = deplex_idx[i_d]
            attn_mask = deplex_mask[i_d - 1]
            seq_len = demulidx.shape[0] // batch_size

            # skip connection
            if (torch.rand(1) > self.skip_dropout).all():
                x = (self.skip_norm_pre(encode_x[i_d]) + self.norm_x(x)) / 2.
            else:
                x = self.norm_x(x)

            deplex_x = self.unpool(
                pooled=x,
                seq_len=seq_len,
                hierarchy_idx=demulidx,
                mask=attn_mask
            )

            # hierarchy attention
            x = st_layer(deplex_x, hierarchy_x, attn_mask, mask_hierarchy=mask_hierarchy)
            decode_x.append(x.clone())

        # return sentence embedding and class token
        return decode_x


class StructuralDocumentProcessor(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ff: int,
            n_blocks: int = 8,
            dropout: float = .1,
            n_xca: int = 2,
            n_hierarchy: int = 3,
            pooler_type: str = "amax",
            ema_weight: float = .99
    ):
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be a positive integer")
        self.d_model = d_model

        if n_blocks < 1:
            raise ValueError("n_blocks must be a positive integer")
        self.n_blocks = n_blocks

        self.sdp_blocks = torch.nn.ModuleList([
                SDPEncoderBlock(
                    d_model=self.d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_hierarchy=n_hierarchy,
                    n_xca=n_xca,
                    pooler_type=pooler_type,
                ) for _ in range(self.n_blocks)
        ])

    def forward(
            self,
            x: torch.Tensor,
            hierarchy_x: torch.Tensor,
            hierarchy: torch.Tensor,
            attention_mask: torch.Tensor | None = None
    ):
        if attention_mask is None:
            attention_mask = torch.zeros(x.shape[:2], device=x.device, dtype=torch.bool)
        x_std = x.std(dim=-1, keepdim=True)
        x = (x - x.mean(dim=-1, keepdim=True)) / torch.where(x_std > 0, x_std, torch.ones_like(x_std))
        init_x = x.clone()
        level_dict, deplex_idx, deplex_hid, deplex_mask, agg_dict = None, None, None, None, None
        for i_block, sdp_block in enumerate(self.sdp_blocks):
            # x is new sentence embedding, hierarchy_x is new class token
            hierarchy_x, level_dict, deplex_idx, deplex_mask, agg_dict = sdp_block(
                init_x,
                hierarchy_x,
                hierarchy,
                attention_mask,
                mask_hierarchy=i_block == 0
            )
        return hierarchy_x, level_dict, deplex_idx, deplex_mask, agg_dict


class StructuralDocumentDecoder(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            seq_len: int,
            d_ff: int,
            n_blocks: int = 1,
            dropout: float = .1,
            n_xca: int = 2,
            n_hierarchy: int = 3,
    ):
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be a positive integer")
        self.d_model = d_model

        if n_blocks < 1:
            raise ValueError("n_blocks must be a positive integer")
        self.n_blocks = n_blocks

        self.sdp_blocks = torch.nn.ModuleList([
                SDPDecoderBlock(
                    d_model=self.d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_hierarchy=n_hierarchy,
                    n_xca=n_xca,
                ) for _ in range(self.n_blocks)
        ])

    def forward(
            self,
            x: torch.Tensor,
            encode_x: Dict[int, torch.Tensor],
            deplex_idx: List[torch.Tensor],
            deplex_mask: List[torch.Tensor],
            hierarchy_x: torch.Tensor,
    ):
        decoded_x = None
        for i_block, sdp_block in enumerate(self.sdp_blocks):
            # x is new sentence embedding, hierarchy_x is new class token
            decoded_x = sdp_block(
                x,
                deplex_idx=deplex_idx,
                deplex_mask=deplex_mask,
                hierarchy_x=hierarchy_x,
                encode_x=encode_x,
                mask_hierarchy=True
            )

        return decoded_x

