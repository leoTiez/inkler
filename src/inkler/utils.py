import subprocess
import torch

from inkler.constants import ROMAN_MAP


def roman_to_int(s: str):
    if len(s) == 0:
        return None
    res = 0
    i = 0
    while i < len(s):
        if s[i] not in ROMAN_MAP:
            return None
        if i + 1 < len(s) and s[i + 1] not in ROMAN_MAP:
            return None

        # if the current value is less than the next value,
        # subtract current from next and add to res
        if i + 1 < len(s) and ROMAN_MAP[s[i]] < ROMAN_MAP[s[i + 1]]:
            res += ROMAN_MAP[s[i + 1]] - ROMAN_MAP[s[i]]

            # skip the next symbol
            i += 1
        else:

            # otherwise, add the current value to res
            res += ROMAN_MAP[s[i]]
        i += 1

    return res


def broadcast_index(index: torch.Tensor, source: torch.Tensor):
    # Ensure index is long
    index = index.long()

    # Add singleton dimensions so it matches source rank
    for _ in range(source.dim() - index.dim()):
        index = index.unsqueeze(-1)

    # Expand to match source shape
    return index.expand_as(source)


def canonical_mask(mask: torch.Tensor, target_type: torch.dtype):
    if not torch.is_floating_point(mask):
        mask = torch.zeros_like(mask, dtype=target_type).masked_fill_(
            mask, float("-inf")
        )
    return mask.to(target_type)


def index_reduce_mps(
        input: torch.Tensor,
        dim: int,
        index: torch.Tensor,
        source: torch.Tensor,
        reduce: str ="sum",
        include_self: bool = False,
        mask: torch.Tensor | None = None,
):
    """
    index_reduce with an optional `mask` selecting which (index, source) pairs
    participate. Entries where `mask` is False are redirected into an extra
    scratch bucket appended along `dim` (instead of compacting them out via
    boolean indexing beforehand, e.g. ``source[mask]``). Boolean-mask
    compaction lowers to `nonzero`, which forces a blocking host<->device
    sync on MPS/CUDA; on long book sequences these syncs dominate wall time
    far more than the actual reduction. Redirecting keeps everything as
    static-shaped ops and is numerically identical to masking first.
    """
    if mask is not None:
        garbage = input.shape[dim]
        pad_shape = list(input.shape)
        pad_shape[dim] = 1
        input = torch.cat([input, input.new_zeros(pad_shape)], dim=dim)
        index = torch.where(mask, index, torch.full_like(index, garbage))

    out = input.clone()
    index_expanded = broadcast_index(index=index, source=source)

    if reduce == "sum":
        out.scatter_add_(dim, index_expanded, source)

    elif reduce == "mean":
        tmp = torch.zeros_like(input)
        count = torch.zeros_like(input)

        tmp.scatter_add_(dim, index_expanded, source)
        count.scatter_add_(dim, index_expanded, torch.ones_like(source))
        count = torch.clamp(count, min=1)

        out = tmp / count

    elif reduce in ("amax", "amin"):
        out.scatter_reduce_(
            dim,
            index_expanded,
            source,
            reduce=reduce,
            include_self=include_self,
        )

    else:
        raise NotImplementedError(f"{reduce} not supported")

    if mask is not None:
        out = out.narrow(dim, 0, garbage)

    return out


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()