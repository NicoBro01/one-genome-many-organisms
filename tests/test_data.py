"""Tests for loading emoji targets (src/nca/data.py) and basic plotting (src/nca/viz.py)."""

import torch

from nca.data import EMOJI, load_target, load_targets
from nca.viz import save_grid, to_rgb


def test_target_shape():
    # 40x40 emoji + 16 px of empty border on each side = 72x72, with 4 channels (RGBA).
    t = load_target("lizard")
    assert t.shape == (4, 72, 72)
    assert t.dtype == torch.float32


def test_target_range():
    # All values must be in [0, 1], the range the network will work in.
    t = load_target("lizard")
    assert t.min() >= 0.0
    assert t.max() <= 1.0


def test_premultiplied():
    # With premultiplied alpha, no colour channel can exceed alpha:
    # an invisible pixel (alpha = 0) must also have zero colour.
    t = load_target("lizard")
    rgb, alpha = t[:3], t[3:4]
    assert (rgb <= alpha + 1e-6).all()


def test_padding_is_empty():
    # The 16-pixel border around the emoji must be completely empty (all zeros).
    t = load_target("lizard")
    assert t[:, :16, :].abs().sum() == 0   # top
    assert t[:, -16:, :].abs().sum() == 0  # bottom
    assert t[:, :, :16].abs().sum() == 0   # left
    assert t[:, :, -16:].abs().sum() == 0  # right


def test_emoji_is_not_empty():
    # Sanity check: the emoji itself must contain some visible pixels.
    t = load_target("lizard")
    assert t[3].sum() > 100


def test_load_targets_stacks_all():
    # Loading several emoji gives one tensor of shape [K, 4, 72, 72].
    names = list(EMOJI)
    ts = load_targets(names)
    assert ts.shape == (len(names), 4, 72, 72)


def test_to_rgb_white_background():
    # A completely empty state (all zeros) must be shown as a white image.
    empty = torch.zeros(4, 8, 8)
    assert torch.allclose(to_rgb(empty), torch.ones(3, 8, 8))


def test_to_rgb_opaque_pixel_keeps_colour():
    # A fully opaque red pixel (alpha = 1) must stay pure red.
    x = torch.zeros(4, 1, 1)
    x[0] = 1.0  # red
    x[3] = 1.0  # alpha
    assert torch.allclose(to_rgb(x)[:, 0, 0], torch.tensor([1.0, 0.0, 0.0]))


def test_save_grid_creates_file(tmp_path):
    # tmp_path is a temporary folder that pytest creates and deletes for us.
    images = torch.rand(3, 3, 10, 10)
    path = tmp_path / "grid.png"
    save_grid(images, path, nrow=2, scale=2)
    assert path.exists()
