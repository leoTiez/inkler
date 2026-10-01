import torch
from math import pi
from torch.distributions.utils import probs_to_logits

from inkler.lexic.abstract_glyph import AbstractGlyph
from inkler.stroke_topology import parents
from inkler.utils import scale_points

DIRECTION = {
    "left": 0,
    "straight": 1,
    "right": 2,
}

PROPERTIES = {
    "x": 0,
    "y": 1,
    "scale": 2,
    "rotation": 3,
}


class WoodlouseInklerGlyph(torch.nn.Module):
    def __init__(self, d_model=384, n_strokes=16):
        super().__init__()
        self.n_strokes = n_strokes
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_model, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, self.n_strokes * (len(DIRECTION) * 4 + len(PROPERTIES)))
        )

    def forward(self, z):
        theta = self.net(z).view(-1, self.n_strokes, (len(DIRECTION) * 4 + len(PROPERTIES)))
        # circle line hook activation
        n_theta = theta.shape[0]
        direct = theta[..., :len(DIRECTION) * 4].reshape(
            n_theta, self.n_strokes, -1, len(DIRECTION)
        ).softmax(dim=-1).reshape(-1, self.n_strokes, len(DIRECTION) * 4)
        properties = theta[..., -len(PROPERTIES):]
        # position
        x = torch.tanh(properties[..., PROPERTIES["x"]]).unsqueeze(-1)
        y = torch.tanh(properties[..., PROPERTIES["y"]]).unsqueeze(-1)
        # Angle
        rotation = torch.tanh(properties[..., PROPERTIES["rotation"]]).unsqueeze(-1)
        # scale
        scale = torch.sigmoid(properties[..., PROPERTIES["scale"]]).unsqueeze(-1)

        return torch.cat([
            direct,
            x,
            y,
            scale,
            rotation
        ], dim=-1)


class WoodlouseCharacterStrokes(AbstractGlyph):
    def __init__(
            self,
            num_points: int = 32,
            margin: float = 0.05,
            min_scale: float = .0,
            noise: float = .05,
            n_noise_knots: int = 6,
            smooth: int = 11,
            **kwargs
    ):
        super().__init__(
            margin=margin,
            noise=noise,
            n_noise_knots=n_noise_knots
        )
        self.num_points = num_points
        self.min_scale = min_scale
        self.register_buffer("smooth", torch.ones(smooth) / float(smooth))
        self.register_buffer("direction_vector", torch.tensor([-1, 0, 1]))

    def _smooth(self, points: torch.Tensor) -> torch.Tensor:
        x_ = torch.nn.functional.pad(
            points,
            (0, 0, self.smooth.shape[0] // 2, self.smooth.shape[0] // 2),
            mode="constant",
        )
        x_ = x_.unfold(dimension=-2, size=self.smooth.shape[0], step=1)
        x_ = x_ @ self.smooth.reshape(1, 1, 1, self.smooth.shape[0], 1)
        return x_.squeeze(-1)

    def draw_line(self, directions: torch.Tensor, scale: torch.Tensor, rotation: torch.Tensor, n_points: int):
        bs, n_strokes = directions.shape[:2]
        t = torch.linspace(0., 1., n_points // 4, device=directions.device)
        t_direct = torch.stack([
            torch.vstack([t, torch.zeros_like(t)]).T,  # along x
            torch.vstack([torch.zeros_like(t), -t]).T,  # along -y
            torch.vstack([-t, torch.zeros_like(t)]).T,  # along -x
            torch.vstack([torch.zeros_like(t), t]).T,   # along y
        ])
        direction_primitives = torch.nn.functional.gumbel_softmax(probs_to_logits(directions), hard=True)
        cum_direct = ((
                             self.direction_vector.reshape(1, 1, 1, -1)
                             * direction_primitives
                     ).sum(-1).cumsum(-2) % 4).int()
        line_primitives = t_direct[cum_direct.flatten()].reshape(
            bs, n_strokes, n_points, 2
        ).cumsum(-2) * scale[..., None, None]
        cos_rot, sin_rot = (pi * rotation).cos(), (pi * rotation).sin()
        rot_mat = torch.stack([
            torch.stack([cos_rot, -sin_rot], dim=-1),
            torch.stack([sin_rot, cos_rot], dim=-1),
        ], dim=-2)
        line_primitives = line_primitives - line_primitives.mean(-2, keepdim=True)
        steps = (rot_mat.unsqueeze(2) @ line_primitives.unsqueeze(-1)).squeeze(-1)
        steps = self._smooth(steps)
        return steps

    def forward(
            self,
            theta: torch.Tensor,
            topology: torch.Tensor,
            add_noise: bool = False,
            n_points: int | None = None ,
            **kwargs
    ):
        """parent_pos=parent_x, child_pos=child_x (B,S). Returns canvas-fitted points (B,S,N,2) in [-1,1]."""
        if n_points is None:
            n_points = self.num_points
        bs, seq_len = theta.shape[:2]
        direct = theta[..., :len(DIRECTION) * 4].reshape(bs, seq_len, -1, len(DIRECTION))
        properties = theta[..., -len(PROPERTIES):]
        parent_pos = properties[..., PROPERTIES["x"]]
        child_pos = properties[..., PROPERTIES["y"]]
        scale = properties[..., PROPERTIES["scale"]]
        scale = scale.masked_fill(scale < self.min_scale, 0.)
        rotation = properties[..., PROPERTIES["rotation"]]

        local_curves, strokes = self.draw_line(direct, n_points=n_points, scale=scale, rotation=rotation), []
        parent_idx = parents(topology)
        for k, parent in enumerate(parent_idx):
            stroke = local_curves[:, k]
            if parent >= 0:
                # interp differentially identifies point in already positioned parent and child stroke
                # -- the difference is added to the child stroke for superosition
                stroke = stroke + (interp(strokes[parent], parent_pos[:, k]) - interp(stroke, child_pos[:, k]))[:, None]
            strokes.append(stroke)
        points = torch.stack(strokes, 1)
        missing_strokes = local_curves.shape[1] - topology.shape[0]
        points = torch.hstack([
            points,
            torch.zeros(
                (points.shape[0], missing_strokes, *points.shape[2:]),
                dtype=points.dtype,
                device=points.device
            )],
        )
        spoints = scale_points(points)
        if add_noise:
            spoints = self._add_noise(spoints, n_points=n_points)
        spoints = scale_points(spoints)
        return spoints * (1 - self.margin)


def interp(polyline, pos):  # point at pos in [-1,1] along polyline (B,N,2) -> (B,2), differentiable in pos
    num_points = polyline.shape[1]
    t = (pos.clamp(-1, 1) + 1) / 2 * (num_points - 1)
    idx = t.floor().long().clamp(max=num_points - 2)
    gather = lambda j: polyline.gather(1, j[:, None, None].expand(-1, 1, 2))[:, 0]
    weight = (t - idx)[:, None]
    return (1 - weight) * gather(idx) + weight * gather(idx + 1)


def woodlouse_contact_loss(points, topology, min_dist=0.06):
    """Penalize strokes i, j that come closer than min_dist (canvas units; 2*width = touching when
    rendered) where topology[i, j] is False. Every point inside the margin is pushed out."""
    batch, num_strokes, num_points, _ = points.shape
    no_interx, no_intery = torch.where((1. - topology).bool())
    sq_point_distance = (points[:, no_interx] - points[:, no_intery]).pow(2).sum(-1)
    dist_strokes = (sq_point_distance + 1e-12).sqrt().amin(-1)
    denom = max(min_dist, 1e-6)
    return (torch.relu(min_dist - dist_strokes) / denom).mean()


