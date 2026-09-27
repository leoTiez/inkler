import torch
import torch.nn.functional as F

from inkler.lexic.abstract_glyph import AbstractGlyph
from inkler.utils import scale_points


PROPERTIES = {
    "x": 0,
    "y": 1,
    "length": 2,
    "angle": 3,
    "curvature": 4,
    "activation": 5,
}

class BezierInklerGlyph(torch.nn.Module):
    """
    Parameterization of a quadratic Bézier curve
    B(t) = (1−t)^2 P0 + 2(1−t)t P1 + t^2P2.
         = P0 + 2t(P1−P0) + t^2(P0−2P1+P2).

    using
    (position, length, angle, width, curvature)
    p(t) = (C−2ℓu)+t(ℓu+4cℓv)−t^2(4cℓv).
    with C=[x y], u=[cosα sinα], v=[−sinα cosα].

    P0 = C − 2ℓu
    P1 = C + 2cℓv.
    P0 = C + 2ℓu
    --> the curvature here represents how P0 and P2 change with respect to control point P1

    """
    def __init__(self, d_model=384, n_strokes=16):
        super().__init__()
        self.n_strokes = n_strokes
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_model, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, n_strokes * 6)
        )

    def forward(self, z):
        theta = self.net(z).view(-1, self.n_strokes, 6)
        # Position in [0,1]
        xy = torch.sigmoid(theta[..., 0:2])
        # Length
        length = torch.sigmoid(theta[..., 2:3])
        # Angle
        angle = torch.tanh(theta[..., 3:4]) * torch.pi
        # Curvature
        curvature = torch.tanh(theta[..., 4:5])
        # activation
        activation = torch.sigmoid(theta[..., 5:6])

        return torch.cat([
            xy,
            length,
            angle,
            curvature,
            activation
        ], dim=-1)


class BezierCharacterStrokes(AbstractGlyph):
    def __init__(
            self,
            n_samples: int = 32,
            margin: float= .05,
            noise: float = .05,
            n_noise_knots: int = 6,
            **kwargs
    ):
        super().__init__(
            margin=margin,
            noise=noise,
            n_noise_knots=n_noise_knots
        )
        self.n_samples = n_samples
        self.register_buffer("t", torch.linspace(0, 1, self.n_samples))

    def forward(self, theta: torch.Tensor, add_noise: bool = False, n_points: int | None = None, **kwargs):
        P0, P1, P2 = bezier_control_points(theta)
        if n_points is None:
            t = self.t
        else:
            t = torch.linspace(0., 1., n_points).to(theta.device)
        points = quadratic_bezier(P0, P1, P2, t)
        spoints = scale_points(points)
        if add_noise:
            spoints = self._add_noise(spoints, n_points=len(t))
        spoints = scale_points(spoints)
        return spoints * (1 - self.margin)


def bezier_control_points(theta: torch.Tensor):
    "Calculate bezier control points from theta parameters"
    x = theta[..., PROPERTIES["x"]]
    y = theta[..., PROPERTIES["y"]]
    length = theta[..., PROPERTIES["length"]]
    angle = theta[..., PROPERTIES["angle"]]
    curvature = theta[..., PROPERTIES["curvature"]]

    C = torch.stack([x, y], dim=-1)
    u = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
    v = torch.stack([-torch.sin(angle), torch.cos(angle)],dim=-1)

    P0 = C - 0.5 * length.unsqueeze(-1) * u
    P1 = C + 2.0 * curvature.unsqueeze(-1) * length.unsqueeze(-1) * v
    P2 = C + 0.5 * length.unsqueeze(-1) * u

    return P0, P1, P2


def quadratic_bezier(P0: torch.Tensor, P1: torch.Tensor, P2: torch.Tensor, t: torch.Tensor):
    "calulate quadratic bezier curve from control points"
    t = t.reshape(1, 1, t.shape[-1], 1)
    return ((1 - t) ** 2 * P0.unsqueeze(-2) + 2 * (1 - t) * t * P1.unsqueeze(-2) + t ** 2 * P2.unsqueeze(-2))


def bezier_contact_loss(points: torch.Tensor, topology: torch.Tensor, min_dist=0.06):
    pair_distance = (points.unsqueeze(1) - points.unsqueeze(2)).pow(2).sum(dim=-1).min(dim=-1).values
    no_penetration = (pair_distance - min_dist).relu()
    penetration = (min_dist - pair_distance).relu()
    if min_dist > 0:
        penetration = penetration / min_dist
    return (topology.unsqueeze(0) * no_penetration + (1. - topology.unsqueeze(0)) * penetration).mean()

