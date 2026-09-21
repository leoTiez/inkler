import subprocess
import torch
import logging
import psutil

from inkler.constants import ROMAN_MAP

logger = logging.getLogger(__name__)


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
        mask: torch.Tensor,
        reduce: str ="sum",
        include_self: bool = False,
        return_idx: bool= True,
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
    garbage = input.shape[dim]
    pad_shape = list(input.shape)
    pad_shape[dim] = 1
    input = torch.cat([input, input.new_zeros(pad_shape)], dim=dim)
    index = torch.where(mask, index, torch.full_like(index, garbage))

    out = input.clone()
    index_expanded = broadcast_index(index=index, source=source)

    out_idx = None
    if reduce == "sum":
        out = out.scatter_add(dim, index_expanded, source)
        out_idx = None

    elif reduce == "mean":
        tmp = torch.zeros_like(input)
        count = torch.zeros_like(input)

        tmp = tmp.scatter_add(dim, index_expanded, source)
        count = count.scatter_add(dim, index_expanded, torch.ones_like(source))
        count = torch.clamp(count, min=1)
        out = tmp / count
        out_idx = None

    elif reduce in ("amax", "amin"):
        if input.device.type == "mps" and torch.are_deterministic_algorithms_enabled():
            out = torch.scatter_reduce(
                out.cpu(),
                dim,
                index_expanded.cpu(),
                source.cpu(),
                reduce=reduce,
                include_self=include_self,
            ).to(input.device)
        else:
            out = torch.scatter_reduce(
                out,
                dim,
                index_expanded,
                source,
                reduce=reduce,
                include_self=include_self,
            )


        if return_idx:
            # fetch max positions
            # remember that it's pooled individually per dimension over the sequence
            # not a single vector selected out of x
            pooled_for_source = out.index_select(0, index)
            is_winner = mask.reshape(-1, 1) & (pooled_for_source == source)

            # Flattened source position in input
            flat_position = torch.arange(index.shape[0], device=source.device, dtype=torch.float32)
            if source.dim() > 1:
                flat_position = flat_position.reshape(-1, 1)
            flat_position = broadcast_index(index=flat_position, source=source)
            invalid_position = float(index.shape[0])

            # get all possible candidate positions --> will select the lowest index over document id hierarchy
            candidates = torch.where(
                is_winner,
                flat_position,
                torch.full_like(flat_position, invalid_position),
            ).to(torch.float32)

            max_indices = torch.full(
                (out.shape[0] - 1, out.shape[1]),
                invalid_position,
                device=source.device,
                dtype=torch.float32,
            )

            # select the lowest index over document id hierarchy
            out_idx, _ = index_reduce_mps(
                input=max_indices,
                dim=0,
                index=index,
                source=candidates,
                reduce="amin",
                include_self=True,
                mask=mask,
                return_idx=False
            )
            out_idx = out_idx.to(torch.int64)

    else:
        raise NotImplementedError(f"{reduce} not supported")

    out = out.narrow(dim, 0, garbage)
    return out, out_idx


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


def log_mps_memory(prefix: str = "", device: str = "mps"):
    if torch.backends.mps.is_available() and "mps" in device.lower():
        allocated = torch.mps.current_allocated_memory() / 2**30
        driver = torch.mps.driver_allocated_memory() / 2**30
        logger.debug(f"{prefix}: MPS tensor={allocated:.2f} GB, driver={driver:.2f} GB")
    elif torch.cuda.is_available() and "cuda" in device.lower():
        allocated = torch.cuda.memory_allocated() / 2 ** 30
        logger.debug(f"{prefix}: CUDA tensor={allocated:.2f} GB")
    else:
        process = psutil.Process()
        allocated = process.memory_info().rss / 2**30
        logger.debug(f"{prefix}: CPU tensor={allocated:.2f} GB")