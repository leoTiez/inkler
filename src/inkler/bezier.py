import torch
import torch.nn.functional as F


def bezier_control_points(theta):
    """
    theta: [B, K, 6]

    Parameters:
        0: x
        1: y
        2: length
        3: angle
        4: curvature
        5: activation

    Returns:
        P0, P1, P2: [B, K, 2]
    """
    x = theta[..., 0]
    y = theta[..., 1]
    length = theta[..., 2]
    angle = theta[..., 3]
    curvature = theta[..., 4]

    C = torch.stack([x, y], dim=-1)
    u = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
    v = torch.stack([-torch.sin(angle), torch.cos(angle)],dim=-1)

    P0 = C - 0.5 * length.unsqueeze(-1) * u
    P1 = C + 2.0 * curvature.unsqueeze(-1) * length.unsqueeze(-1) * v
    P2 = C + 0.5 * length.unsqueeze(-1) * u

    return P0, P1, P2


def quadratic_bezier(P0, P1, P2, t):
    """
    P0, P1, P2: [..., 2]
    t:           [...]
    """

    t = t.unsqueeze(-1)
    return ((1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2)


def bezier_pair_distance(
    P0_i, P1_i, P2_i,
    P0_j, P1_j, P2_j,
    n_samples=32,
):
    """
    Estimate minimum distance between two quadratic
    Bézier curves.

    No rendering involved.

    Returns:
        [B]
    """

    device = P0_i.device

    t = torch.linspace(0.0,1.0, n_samples, device=device)

    # Curve i
    Pi = quadratic_bezier(
        P0_i[:, None, :],
        P1_i[:, None, :],
        P2_i[:, None, :],
        t[None, :],
    )
    # [B, T, 2]

    # Curve j
    Pj = quadratic_bezier(
        P0_j[:, None, :],
        P1_j[:, None, :],
        P2_j[:, None, :],
        t[None, :],
    )
    # [B, T, 2]

    # Pairwise distances between sampled points.
    d2 = (
        Pi[:, :, None, :]
        - Pj[:, None, :, :]
    ).pow(2).sum(dim=-1)

    min_d2 = d2.flatten(1).min(dim=1).values

    return torch.sqrt(min_d2 + 1e-12)


def contact_loss(
    theta,
    allowed_matrix,
    contact_threshold,
    n_samples=16,
):
    """
    theta:
        [B, K, 6]

    allowed_matrix:
        [K, K]

        1 = intersection/contact allowed
        0 = intersection/contact forbidden

    contact_threshold:
        Usually stroke_width for physical stroke overlap.
        Use a small epsilon for centerline intersection.

    Returns:
        scalar loss
    """

    B, K, _ = theta.shape
    P0, P1, P2 = bezier_control_points(theta)
    loss = torch.zeros((), device=theta.device,)
    bezier_curves = quadratic_bezier(
        P0.unsqueeze(-2),
        P1.unsqueeze(-2),
        P2.unsqueeze(-2),
        torch.linspace(0.0,1.0, n_samples, device=theta.device).unsqueeze(0)
    )
    pair_distance = (bezier_curves.unsqueeze(1) - bezier_curves.unsqueeze(2)).pow(2).sum(dim=-1).min(dim=-1).values
    no_penetration = (pair_distance - contact_threshold).relu()
    penetration = (contact_threshold - pair_distance).relu()
    if contact_threshold > 0:
        penetration = penetration / contact_threshold
    return (allowed_matrix.unsqueeze(0) * no_penetration + (1. - allowed_matrix.unsqueeze(0)) * penetration).mean()



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
        # # Width
        # width = torch.sigmoid(theta[..., 4:5])
        # Curvature
        curvature = torch.tanh(theta[..., 4:5])
        # activation
        activation = torch.sigmoid(theta[..., 5:6])

        return torch.cat([
            xy,
            length,
            angle,
            # width,
            curvature,
            activation
        ], dim=-1)


class BezierGlyphDecoder(torch.nn.Module):
    def __init__(self, n_strokes=16, d_model=384, act_thresh: float = .5):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_strokes * 6, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, d_model)
        )
        self.act_thresh = act_thresh

    def forward(self, theta):
        return self.net(theta.flatten(1))