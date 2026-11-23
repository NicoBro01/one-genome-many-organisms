# One Genome, Many Organisms

Neural Cellular Automata (NCA) with a heritable genome. A single learned growth
rule, run by every cell of a grid, grows a whole organism (an emoji) from one
seed cell and regenerates it after damage. A small genome written into the seed
decides *which* organism grows, which lets us breed hybrids, mutants and chimeras.

Project for the *Deep Learning and Applied AI* exam (Sapienza, a.y. 2025/2026).
Areas: artificial life, image generation.

- Design: [`docs/design.md`](docs/design.md)
- Plan: [`docs/plan.md`](docs/plan.md)

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync              # create .venv and install the exact dependency versions
uv run pytest -q     # run the tests
```

Training uses the Apple-silicon GPU (MPS) when available, the CPU otherwise.

## Status

Work in progress.
