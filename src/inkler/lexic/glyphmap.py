import torch
from math import pi
from torch.distributions.utils import probs_to_logits

from inkler.lexic.abstract_glyph import AbstractGlyph
from inkler.stroke_topology import parents
from inkler.utils import scale_points

LEXIC = {
    "circle": 0,
    "line": 1,
    "curve": 2,
}

PROPERTIES = {
    "x": 0,
    "y": 1,
    "scale": 2,
    "rotation": 3,
}

# HOOK = torch.tensor([
#     [-0.75, -0.75],
#     [ 0.45, -0.75],
#     [ 0.75, -0.60],
#     [ 0.75,  0.05],
#     [ 0.72,  0.40],
#     [ 0.52,  0.68],
#     [ 0.20,  0.82],
#     [-0.15,  0.80],
#     [-0.45,  0.62],
#     [-0.62,  0.38],
# ], dtype=torch.float32)

HOOK = torch.tensor([
    [-0.75, -0.22],
    [-0.63, -0.06],
    [-0.48,  0.07],
    [-0.30,  0.16],
    [-0.10,  0.21],
    [ 0.10,  0.21],
    [ 0.30,  0.16],
    [ 0.48,  0.07],
    [ 0.63, -0.06],
    [ 0.75, -0.22],
], dtype=torch.float32)


class LexicInklerGlyph(torch.nn.Module):
    def __init__(self, d_model=384, n_strokes=16):
        super().__init__()
        self.n_strokes = n_strokes
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_model, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, self.n_strokes * (len(LEXIC) + len(PROPERTIES)))
        )
        self.temperature = 1.0

    def forward(self, z):
        theta = self.net(z).view(-1, self.n_strokes, (len(LEXIC) + len(PROPERTIES)))
        # circle line hook activation
        clh = (theta[..., :len(LEXIC)] / self.temperature).softmax(dim=-1)
        properties = theta[..., len(LEXIC):]
        # position
        x = torch.tanh(properties[..., PROPERTIES["x"]]).unsqueeze(-1)
        y = torch.tanh(properties[..., PROPERTIES["y"]]).unsqueeze(-1)
        # Angle
        rotation = torch.tanh(properties[..., PROPERTIES["rotation"]]).unsqueeze(-1)
        # scale
        scale = torch.sigmoid(properties[..., PROPERTIES["scale"]]).unsqueeze(-1)

        return torch.cat([
            clh,
            x,
            y,
            scale,
            rotation
        ], dim=-1)


class LexicCharacterStrokes(AbstractGlyph):
    def __init__(
            self,
            num_points: int = 32,
            margin: float = 0.05,
            min_scale: float = .0,
            noise: float = .02,
            n_noise_knots: int = 6,
            snap: float = -.0,
            **kwargs
    ):
        super().__init__(
            margin=margin,
            noise=noise,
            n_noise_knots=n_noise_knots
        )
        self.num_points = num_points
        self.min_scale = min_scale
        self.snap = snap

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
        primitive_probs = theta[..., :len(LEXIC)]
        properties = theta[..., len(LEXIC):]
        parent_pos = properties[..., PROPERTIES["x"]]
        child_pos = properties[..., PROPERTIES["y"]]
        scale = properties[..., PROPERTIES["scale"]]
        scale = scale.masked_fill(scale < self.min_scale, 0.)
        rotation = properties[..., PROPERTIES["rotation"]]
        parent_idx = parents(topology)
        local_curves, strokes = shapes(primitive_probs, scale, rotation, n_points, snap=self.snap), []
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


def hook_profile(num_points, device=None):
    """Heading of HOOK at uniform arc-length segment midpoints, plus its total arc length."""
    segments = (HOOK[1:] - HOOK[:-1]).to(device)
    angle = torch.atan2(segments[:, 1], segments[:, 0])
    angle = angle[0] + torch.cat([angle[:1] * 0, (angle.diff() + pi) % (2 * pi) - pi]).cumsum(0)  # unwrap
    arc = segments.norm(dim=-1).cumsum(0)
    mid = (torch.arange(num_points - 1, device=device) + 0.5) / (num_points - 1) * arc[-1]
    return angle[torch.searchsorted(arc, mid).clamp(max=len(angle) - 1)], arc[-1]


def shapes(primitive_probs, scale, rotation, num_points=64, snap: float =0.0):
    """primitive_probs (B,S,3) [circle,line,hook], scale, rotation (B,S) -> local curves (B,S,N,2).
    Each primitive = heading profile + arc length; blending them is exact for one-hot probs."""
    hard = torch.nn.functional.one_hot(primitive_probs.argmax(-1), primitive_probs.shape[-1]).to(primitive_probs.dtype)
    primitive_probs = primitive_probs + snap * (hard - primitive_probs).detach()
    u = torch.linspace(-1, 1, num_points, device=primitive_probs.device)
    u_mid = (u[1:] + u[:-1]) / 2
    hook_heading, hook_length = hook_profile(num_points, u.device)
    heading = torch.stack([pi * (u_mid + 1), 0 * u_mid, hook_heading])      # (3, N-1)
    lengths = torch.stack([u.new_tensor(2 * pi), u.new_tensor(2.), hook_length])  # circle radius s, line 2s, hook
    length = (primitive_probs @ lengths) * scale
    phi = torch.einsum('bsp,pn->bsn', primitive_probs, heading) + pi * rotation[..., None]
    steps = torch.stack([phi.cos(), phi.sin()], -1) * (length / (num_points - 1))[..., None, None]
    return torch.nn.functional.pad(steps.cumsum(-2), (0, 0, 1, 0))


def interp(polyline, pos):  # point at pos in [-1,1] along polyline (B,N,2) -> (B,2), differentiable in pos
    num_points = polyline.shape[1]
    t = (pos.clamp(-1, 1) + 1) / 2 * (num_points - 1)
    idx = t.floor().long().clamp(max=num_points - 2)
    gather = lambda j: polyline.gather(1, j[:, None, None].expand(-1, 1, 2))[:, 0]
    weight = (t - idx)[:, None]
    return (1 - weight) * gather(idx) + weight * gather(idx + 1)


def lexic_contact_loss(points, topology, min_dist=0.06):
    """Penalize strokes i, j that come closer than min_dist (canvas units; 2*width = touching when
    rendered) where topology[i, j] is False. Every point inside the margin is pushed out."""
    batch, num_strokes, num_points, _ = points.shape
    no_interx, no_intery = torch.where((1. - topology).bool())
    sq_point_distance = (points[:, no_interx] - points[:, no_intery]).pow(2).sum(-1)
    dist_strokes = (sq_point_distance + 1e-12).sqrt().amin(-1)
    denom = max(min_dist, 1e-6)
    return (torch.relu(min_dist - dist_strokes) / denom).mean()


