import numpy as np
from typing import Tuple, Dict, List
import torch

from inkler.modules import XCA, AttentionWrapper, Pooler, Unpooler
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
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], List[torch.Tensor], Dict[int, torch.Tensor]]:
        out_embedding = {}
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
            deplex_idx.append(demu_idx.clone())
            deplex_mask[i_d] = attention_mask_.clone()
            out_embedding[i_d] = x.clone()
            # adapt hierarchy ids to new layout
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
        return x, out_embedding, deplex_idx, deplex_mask


class SDPDecoderBlock(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ff: int,
            dropout: float = .1,
            n_xca: int = 2,
            n_hierarchy: int = 3,
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

    def forward(
            self,
            x: torch.Tensor,
            deplex_idx: List[torch.Tensor],
            deplex_mask: List[torch.Tensor],
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

            deplex_x = self.unpool(
                pooled=x,
                seq_len=seq_len,
                # max_indices=demulidx,
                hierarchy_idx=demulidx,
                # mask=demulmask
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

        self.ema_sentence_set = False
        self.ema_document_set = False
        self.register_buffer("ema_sentence", torch.zeros((1, 1, self.d_model)))
        self.register_buffer("ema_document", torch.zeros((1, 1, self.d_model)))
        self.ema_weight = ema_weight
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

    @staticmethod
    def _update_ema(x: torch.Tensor, ema: torch.Tensor, mask: torch.Tensor, ema_weight: float):
        feat_dim = x.shape[-1]
        return ema * ema_weight + (1 - ema_weight) * x[~mask].reshape(-1, feat_dim).mean(0, keepdim=True).unsqueeze(0)

    @torch.no_grad()
    def _update_sentence_ema(self, x: torch.Tensor, mask: torch.Tensor):
        if self.training:
            ema_weight = self.ema_weight if self.ema_sentence_set else 0.
            self.ema_sentence_set = True
            self.ema_sentence = self._update_ema(x, self.ema_sentence, mask, ema_weight)

    @torch.no_grad()
    def _update_document_ema(self, x: torch.Tensor, mask: torch.Tensor):
        if self.training:
            ema_weight = self.ema_weight if self.ema_document_set else 0.
            self.ema_document_set = True
            self.ema_document = self._update_ema(x, self.ema_document, mask, ema_weight)

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
        self._update_sentence_ema(x, attention_mask.bool())
        x = x - self.ema_sentence
        level_dict, deplex_idx, deplex_hid, deplex_mask = None, None, None, None
        for i_block, sdp_block in enumerate(self.sdp_blocks):
            # x is new sentence embedding, hierarchy_x is new class token
            hierarchy_x, level_dict, deplex_idx, deplex_mask = sdp_block(
                x,
                hierarchy_x,
                hierarchy,
                attention_mask,
                mask_hierarchy=i_block == 0
            )

        level_dict[-1] = x.clone()
        document_key = max(level_dict.keys())
        batch_size = hierarchy_x.shape[0]
        self._update_document_ema(hierarchy_x, torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device))
        hierarchy_x = hierarchy_x - self.ema_document
        level_dict[document_key] = hierarchy_x.clone()
        return hierarchy_x, level_dict, deplex_idx, deplex_mask


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
            x: Dict[int, torch.Tensor],
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
                mask_hierarchy=True
            )

        return decoded_x

