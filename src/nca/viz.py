"""Turning NCA states into pictures."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def to_rgb(x: torch.Tensor) -> torch.Tensor:
    """Convert a (premultiplied) RGBA state into an RGB image on a white background.

    Works on a single state [C, H, W] or a batch [N, C, H, W]; only the first four
    channels (RGBA) are used, so it also accepts full 16-channel NCA states.

    Compositing over white with premultiplied colours is simply
        rgb_out = rgb + (1 - alpha) * white = rgb + 1 - alpha
    """
    rgb = x[..., :3, :, :]
    alpha = x[..., 3:4, :, :].clamp(0.0, 1.0)
    return (rgb + 1.0 - alpha).clamp(0.0, 1.0)


def save_grid(images: torch.Tensor, path: Path, nrow: int = 4, scale: int = 4) -> None:
    """Save a batch of RGB images [N, 3, H, W] (values in [0, 1]) as one PNG grid.

    Images are placed in rows of `nrow`, separated by a 2-pixel white gap, and
    enlarged `scale` times with nearest-neighbour upsampling so each cell of the
    automaton stays a crisp square.
    """
    images = images.detach().cpu().clamp(0.0, 1.0)
    n, _, h, w = images.shape
    ncol = min(nrow, n)
    nrows = (n + ncol - 1) // ncol  # ceiling division
    gap = 2

    # Start from a white canvas and paste each image into its slot.
    canvas = torch.ones(3, nrows * (h + gap) + gap, ncol * (w + gap) + gap)
    for i in range(n):
        r, c = divmod(i, ncol)
        top, left = gap + r * (h + gap), gap + c * (w + gap)
        canvas[:, top:top + h, left:left + w] = images[i]

    # [3, H, W] floats -> [H, W, 3] bytes, the layout PIL expects.
    array = (canvas.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    img = Image.fromarray(array)
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
