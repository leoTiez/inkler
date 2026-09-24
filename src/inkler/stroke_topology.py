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
def get_stroke_topology(n_strokes: int = 16, n_connect: int = 4, use_connector_stroke: bool = False):
    n_independent = n_strokes // 4

    block_mat = create_block_matrix(n_strokes - n_independent, n_connect)

    topology_mat = torch.eye(n_strokes, dtype=torch.int)
    topology_mat[:-n_independent, :-n_independent] = block_mat
    if use_connector_stroke:
        topology_mat[:-n_independent, 0] = 1.
        topology_mat[0, :-n_independent] = 1.
    return topology_mat
