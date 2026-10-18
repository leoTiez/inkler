import torch


class GlyphRenderer(torch.nn.Module):
    def __init__(self, res: int = 128, width: float = 0.03, blur: float  = 0.01, **kwargs):
        super().__init__()
        self.res = res
        self.width = width
        self.blur = blur

    def forward(self, points: torch.Tensor):
        """Soft rasterizer: points (B,S,N,2) -> ink (B,res,res). Small blur ~ hard render."""
        if points.dim() > 4:
            batch_size, seq_len = points.shape[:2]
            points = points.reshape(batch_size * seq_len, *points.shape[2:])
        grid = torch.linspace(-1, 1, self.res, device=points.device)
        pixels = torch.stack(torch.meshgrid(grid, -grid, indexing='xy'), -1).reshape(-1, 2)
        seg_start = points[:, :, :-1].flatten(1, 2)
        seg_dir = (points[:, :, 1:] - points[:, :, :-1]).flatten(1, 2)
        offset = pixels[None, :, None] - seg_start[:, None]  # (B,R,M,2)
        t = ((offset * seg_dir[:, None]).sum(-1) / (seg_dir.pow(2).sum(-1)[:, None] + 1e-9)).clamp(0, 1)
        dist = ((offset - t[..., None] * seg_dir[:, None]).pow(2).sum(-1).amin(-1) + 1e-12).sqrt()
        return torch.sigmoid((self.width - dist) / self.blur).view(-1, self.res, self.res)