import torch
from inkler.lexic.abstract_glyph import AbstractGlyph
from inkler.utils import scale_points
from torchdiffeq import odeint_adjoint

N_PARAMETERS = 2 * 3


class ODEInklerGlyph(torch.nn.Module):
    def __init__(self, d_model=384, n_strokes=16):
        super().__init__()
        self.n_strokes = n_strokes
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_model, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, n_strokes * (2 * N_PARAMETERS + 2))
        )

    def forward(self, z):
        theta = self.net(z).view(-1, self.n_strokes, (2 * N_PARAMETERS + 2))
        # Position in [0,1]
        xy = torch.sigmoid(theta[..., :2])
        # params
        parameters = theta[..., 2:]

        return torch.cat([
            xy,
            parameters,
        ], dim=-1)


class MLPPropagator(torch.nn.Module):
    def __init__(self, W1: torch.Tensor, W2: torch.Tensor):
        super().__init__()
        self.W1 = W1.reshape(W1.shape[0], W2.shape[1], N_PARAMETERS // 2, -1)
        self.W2 = W2.reshape(W1.shape[0], W2.shape[1], N_PARAMETERS // 2, -1)

    def forward(self, t, x):
        if self.W1 is None or self.W2 is None:
            raise ValueError("MLPPropagator hasn't been initialized")
        z = torch.einsum("...c,...dc", x, self.W1)
        z = z.relu()
        return torch.einsum("...cd,...c", self.W2, z)


class ODECharacterStrokes(AbstractGlyph):
    def __init__(
            self,
            n_samples: int = 32,
            margin: float= .05,
            noise: float = .05,
            n_noise_knots: int = 6,
            method: str= "dopri5",
            **kwargs
    ):
        super().__init__(
            margin=margin,
            noise=noise,
            n_noise_knots=n_noise_knots
        )
        self.method = method
        self.n_samples = n_samples
        self.register_buffer("t", torch.linspace(0, 1, self.n_samples))

    def forward(self, theta: torch.Tensor, add_noise: bool = False, n_points: int | None = None, **kwargs):
        init_xy = theta[..., :2]
        W1 = theta[..., 2:N_PARAMETERS + 2]
        W2 = theta[..., N_PARAMETERS + 2:]
        propagator = MLPPropagator(W1, W2)

        if n_points is None:
            t = self.t
        else:
            t = torch.linspace(0., 1., n_points).to(theta.device)

        points = odeint_adjoint(
            func=propagator,
            y0=init_xy,
            t=t,
            method=self.method,
        ).permute(1, 2, 0, -1)
        spoints = scale_points(points)
        if add_noise:
            spoints = self._add_noise(spoints, n_points=len(t))
        spoints = scale_points(spoints)
        return spoints * (1 - self.margin)


def ode_contact_loss(points: torch.Tensor, topology: torch.Tensor, min_dist=0.06):
    pair_distance = (points.unsqueeze(1) - points.unsqueeze(2)).pow(2).sum(dim=-1).min(dim=-1).values
    no_penetration = (pair_distance - min_dist).relu()
    penetration = (min_dist - pair_distance).relu()
    if min_dist > 0:
        penetration = penetration / min_dist
    return (topology.unsqueeze(0) * no_penetration + (1. - topology.unsqueeze(0)) * penetration).mean()