from abc import ABC
import torch


class AbstractGlyph(torch.nn.Module, ABC):
    def __init__(
            self,
            margin: float = 0.05,
            noise: float = .05,
            n_noise_knots: int = 6,
    ):
        super().__init__()
        self.margin = margin
        self.noise = noise
        self.n_noise_knots = n_noise_knots

    def _add_noise(self, points: torch.Tensor, n_points: int) -> torch.Tensor:
        batch, num_strokes = points.shape[:2]
        knots = torch.randn(batch * num_strokes, 2, self.n_noise_knots, device=points.device) * self.noise
        knots = torch.cat([knots, knots[..., :1]], -1)  # same offset at both ends: circles stay closed
        wobble = torch.nn.functional.interpolate(knots, size=n_points, mode='linear', align_corners=True)
        points = points + wobble.view(
            batch,
            num_strokes,
            2,
            n_points
        ).transpose(-1, -2)
        return points