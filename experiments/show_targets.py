"""Save a picture of the target organisms, exactly as the NCA will see them.

Run from the repository root:
    uv run python experiments/show_targets.py
"""

from pathlib import Path

from nca.data import EMOJI, load_targets
from nca.viz import save_grid, to_rgb

names = list(EMOJI)
targets = load_targets(names)  # [K, 4, 72, 72], premultiplied RGBA
print(f"Loaded {len(names)} targets {names}, tensor shape {tuple(targets.shape)}")

# Fraction of "alive" pixels (alpha > 0.1) in each 72x72 grid.
for name, t in zip(names, targets):
    alive = (t[3] > 0.1).float().mean().item()
    print(f"  {name:10s} alive pixels: {alive:.1%}")

out = Path("figures/targets.png")
save_grid(to_rgb(targets), out, nrow=len(names), scale=4)
print(f"Saved {out}")
