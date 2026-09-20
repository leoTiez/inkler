import torch
import torch.nn as nn
import torch.nn.functional as F
from networkx.algorithms import threshold


class DifferentiableGlyphRenderer(nn.Module):
    """
    Differentiable renderer for a collection of curved strokes.

    Each stroke has 7 parameters:

        [x, y, length, angle, width, curvature, activation]

    All coordinates are normalized to [0, 1].

    Input:
        theta: [B, K, 7]

    Output:
        image: [B, 1, H, W]

    Everything is differentiable with respect to theta.
    """

    def __init__(
            self,
            image_size=64,
            n_samples=64,
            max_width=0.08,
            min_width=0.001,
            max_length=1.0,
            raster_temperature=0.002,
            margin: float = .1,
            threshold: float| None = 0.5,
    ):
        super().__init__()

        self.image_size = image_size
        self.n_samples = n_samples
        self.max_width = max_width
        self.min_width = min_width
        self.max_length = max_length
        self.raster_temperature = raster_temperature

        # Pixel coordinates in normalized [0,1] space.
        y, x = torch.meshgrid(
            torch.linspace(0, 1, image_size),
            torch.linspace(0, 1, image_size),
            indexing="ij",
        )

        # [H, W]
        self.register_buffer("grid_x", x)
        self.register_buffer("grid_y", y)

        # Parameter along each stroke.
        # [S]
        t = torch.linspace(
            0.0,
            1.0,
            n_samples
        )
        self.margin = margin
        self.threshold = threshold
        self.register_buffer("t", t)

    def stroke_centerline(self, theta):
        """
        Convert stroke parameters to sampled points.

        theta:
            [B, K, 7]

        returns:
            points:
                [B, K, S, 2]
        """

        x = theta[..., 0]
        y = theta[..., 1]
        length = theta[..., 2]
        angle = theta[..., 3]
        curvature = theta[..., 5]

        # --------------------------------------------------
        # Orientation vectors
        # --------------------------------------------------

        ux = torch.cos(angle)
        uy = torch.sin(angle)

        # Perpendicular direction
        vx = -uy
        vy = ux

        # --------------------------------------------------
        # Parameter t
        # --------------------------------------------------

        t = self.t.to(theta.device)

        # reshape for broadcasting:
        # [1, 1, S]
        t = t.view(1, 1, -1)

        # --------------------------------------------------
        # Centered straight-line component
        #
        # t = 0     -> -length/2
        # t = 0.5   -> 0
        # t = 1     -> +length/2
        # --------------------------------------------------

        longitudinal = length.unsqueeze(-1) * (t - 0.5)

        # --------------------------------------------------
        # Curvature
        #
        # 4t(1-t) is:
        #
        # 0 at endpoints
        # 1 at center
        #
        # Therefore curvature controls the maximum
        # perpendicular displacement.
        # --------------------------------------------------

        transverse = (
            4.0
            * curvature.unsqueeze(-1)
            * length.unsqueeze(-1)
            * t
            * (1.0 - t)
        )

        # --------------------------------------------------
        # Construct x/y coordinates
        # --------------------------------------------------

        px = (
            x.unsqueeze(-1)
            + longitudinal * ux.unsqueeze(-1)
            + transverse * vx.unsqueeze(-1)
        )

        py = (
            y.unsqueeze(-1)
            + longitudinal * uy.unsqueeze(-1)
            + transverse * vy.unsqueeze(-1)
        )

        return torch.stack([px, py], dim=-1)

    def render_single_stroke(
        self,
        points,
        width,
    ):
        """
        Differentiably rasterize one stroke.

        points:
            [B, S, 2]

        width:
            [B, 1]

        returns:
            [B, H, W]
        """

        B = points.shape[0]

        # --------------------------------------------------
        # Pixel grid
        # --------------------------------------------------

        gx = self.grid_x.to(points.device)
        gy = self.grid_y.to(points.device)

        # [1, H, W]
        gx = gx.unsqueeze(0)
        gy = gy.unsqueeze(0)

        # --------------------------------------------------
        # Distance from every pixel to every sampled
        # centerline point.
        #
        # [B, H, W, S]
        # --------------------------------------------------

        px = points[..., 0]  # [B, S]
        py = points[..., 1]

        dx = gx.unsqueeze(-1) - px[:, None, None, :]
        dy = gy.unsqueeze(-1) - py[:, None, None, :]

        d2 = dx**2 + dy**2

        # --------------------------------------------------
        # Soft minimum over centerline samples.
        #
        # This approximates:
        #
        # min_s ||pixel - p_s||²
        #
        # while remaining differentiable.
        # --------------------------------------------------

        tau = self.raster_temperature

        soft_d2 = -tau * torch.logsumexp(
            -d2 / tau,
            dim=-1,
        )

        soft_d2 = soft_d2.clamp_min(0.0)

        # --------------------------------------------------
        # Convert distance to soft coverage.
        #
        # Pixels inside the stroke become ~1.
        # Pixels outside become ~0.
        # --------------------------------------------------

        radius = width.squeeze(-1) / 2.0

        # [B, 1, 1]
        radius2 = radius[:, None, None] ** 2

        # Width-dependent antialiasing.
        softness = (
            0.25 * radius2
            + 1e-5
        )

        alpha = torch.sigmoid(
            (radius2 - soft_d2) / softness
        )
        if threshold is not None:
            alpha = (alpha > 0.5).float()
        return alpha

    def forward(self, theta):
        """
        Render all strokes.

        theta:
            [B, K, 7]

        returns:
            glyph:
                [B, H, W]
        """

        # --------------------------------------------------
        # Construct centerlines
        # --------------------------------------------------
        batch_size = theta.shape[0]
        points = self.stroke_centerline(theta)

        # rescale to the same canvas size and add margin
        minv = points.transpose(1, -1).flatten(2).min(-1, keepdim=True)[0].reshape(batch_size, 1, 1, -1)
        maxv = points.transpose(1, -1).flatten(2).max(-1, keepdim=True)[0].reshape(batch_size, 1, 1, -1)
        points = self.margin + (1. - 2 * self.margin) * (points - minv) / (maxv - minv)
        # --------------------------------------------------
        # Width
        #
        # theta[..., 4] is assumed to already be in [0,1].
        # --------------------------------------------------

        width_parameter = theta[..., 4].relu()
        active = (width_parameter > 0).float()

        width = (
            self.min_width * active
            + width_parameter
            * (
                self.max_width
                - self.min_width
            )
        )

        # --------------------------------------------------
        # Render every stroke
        # --------------------------------------------------

        stroke_images = []

        K = theta.shape[1]

        for k in range(K):

            stroke = self.render_single_stroke(
                points[:, k],
                width[:, k:k+1],
            )

            stroke_images.append(stroke)

        # [B, K, H, W]
        strokes = torch.stack(
            stroke_images,
            dim=1
        )

        # --------------------------------------------------
        # Alpha compositing.
        #
        # Probability that at least one stroke covers
        # a pixel:
        #
        # 1 - product(1-alpha_k)
        # --------------------------------------------------

        glyph = 1.0 - torch.prod(
            1.0 - strokes.clamp(0, 1),
            dim=1
        )

        return glyph