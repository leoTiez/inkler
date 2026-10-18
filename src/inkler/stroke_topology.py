import torch


@torch.no_grad()
def create_block_matrix(n_strokes: int, n_connect: int):
    block_mat = torch.zeros((n_strokes, n_strokes), dtype=torch.int)
    # Calculate step size between block origins (overlap of 1 means step = block_size - 1)
    step = n_connect - 1
    # Number of valid submatrices that fit along the diagonal
    num_blocks = (n_strokes - n_connect) // step + 1 if n_strokes >= n_connect else 0

    if num_blocks > 0:
        # Create sliding windows along both dimensions
        blocks = block_mat.unfold(0, n_connect, step).unfold(1, n_connect, step)

        # Fill the diagonal blocks with 1s
        # blocks shape: [num_blocks_row, num_blocks_col, block_size, block_size]
        blocks[torch.arange(num_blocks), torch.arange(num_blocks)] = 1
    return block_mat


@torch.no_grad()
def keep_first_true_per_row(matrix: torch.Tensor) -> torch.Tensor:
    row_indices = torch.where(matrix > 0)[0]
    first_true_indices = matrix.float().argmax(dim=1)
    result = torch.zeros_like(matrix)
    result[row_indices, first_true_indices[row_indices]] = 1

    return result


@torch.no_grad()
def get_stroke_topology(
        n_strokes: int = 16,
        n_block: int = 0,
        use_connector_stroke: bool = False,
        n_connect: int = 5,
        symmetric: bool = False,
) -> torch.Tensor:
    if n_block < 2:
        topology_mat = torch.triu(torch.ones_like(torch.eye(n_strokes, dtype=torch.int)), diagonal=-n_connect)
    else:
        n_independent = n_block - 1
        if use_connector_stroke:
            block_mat = create_block_matrix(n_strokes - n_independent, 2)
        else:
            block_mat = create_block_matrix(n_strokes - n_independent, n_block)

        no_block = torch.where(~block_mat.any(dim=1))[0]
        block_mat[no_block, torch.maximum(no_block - 1, torch.tensor(0))] = 1
        topology_mat = torch.eye(n_strokes, dtype=torch.int)
        topology_mat[:-n_independent, :-n_independent] = block_mat
        if use_connector_stroke:
            topology_mat[:n_connect, 0] = 1.
            topology_mat[0, :n_connect] = 1.

    topology_mat = keep_first_true_per_row(topology_mat)
    if symmetric:
        topology_mat = (topology_mat + topology_mat.T + torch.eye(n_strokes, dtype=torch.int)).clip(min=0., max=1.)
    return topology_mat


# requires gradient
def parents(topology):  # topology: (S,S) bool -> parent index per stroke (-1 = root)
    return [max([j for j in range(k) if topology[k][j]], default=-1) for k in range(len(topology))]


