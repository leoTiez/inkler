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
):
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

    return out