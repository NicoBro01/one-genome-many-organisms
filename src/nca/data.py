"""Loading emoji images as training targets for the NCA.

A target is the image the cellular automaton must learn to grow. We use emoji
from Google's Noto Emoji set (Apache 2.0 licence): they are small, colourful and
have a transparent background, so their alpha channel tells us exactly where
the "organism" is.
"""

import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Folder where the downloaded PNGs live: <repo>/data/emoji.
# Path(__file__) is this file (src/nca/data.py); parents[2] goes up three levels
# to the repository root, so the path works whatever folder we run Python from.
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "emoji"

# Our organisms: a readable name -> the emoji's Unicode code point (in hex),
# which is how the Noto repository names its files.
EMOJI = {
    "lizard": "1f98e",     # 🦎
    "butterfly": "1f98b",  # 🦋
    "mushroom": "1f344",   # 🍄
    "octopus": "1f419",    # 🐙
}

# We use the flat "2D" style (the repository also has a shaded "3D" style, which
# would be harder to learn).
NOTO_URL = "https://raw.githubusercontent.com/googlefonts/noto-emoji/main/2D/png/128/emoji_u{code}.png"


def download_emoji(name: str, dest: Path = DATA_DIR) -> Path:
    """Download the 128x128 PNG of an emoji (only if we do not have it yet)."""
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{name}.png"
    if not path.exists():
        url = NOTO_URL.format(code=EMOJI[name])
        urllib.request.urlretrieve(url, path)
    return path


def load_target(name: str, size: int = 40, pad: int = 16) -> torch.Tensor:
    """Load an emoji as a premultiplied RGBA tensor of shape [4, size + 2*pad, size + 2*pad].

    Steps:
      1. open the PNG and make sure it has 4 channels (RGBA);
      2. shrink it to size x size pixels;
      3. convert to floats in [0, 1] with shape [4, H, W];
      4. premultiply the colours by alpha;
      5. add an empty border of `pad` pixels on every side.
    """
    path = download_emoji(name)

    # 1-2. Open, force RGBA, resize. LANCZOS is a high-quality resampling filter
    # that keeps edges smooth when shrinking from 128 px to 40 px.
    img = Image.open(path).convert("RGBA").resize((size, size), Image.LANCZOS)

    # 3. PIL gives integers 0..255 with shape [H, W, 4]. We divide by 255 to get
    # floats in [0, 1] and move the channel axis first ([4, H, W]), the layout
    # PyTorch convolutions expect.
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    x = x.permute(2, 0, 1)

    # 4. Premultiply: RGB *= alpha. Invisible pixels become exactly [0, 0, 0, 0].
    x[:3] = x[:3] * x[3:4]

    # 5. Pad with zeros: (left, right, top, bottom) for the last two dimensions.
    x = F.pad(x, (pad, pad, pad, pad))
    return x


def load_targets(names: list[str], size: int = 40, pad: int = 16) -> torch.Tensor:
    """Load several emoji at once and stack them into a tensor of shape [K, 4, H, W]."""
    return torch.stack([load_target(n, size, pad) for n in names])
