"""WeVisOne-VL Data Engine (WDE) MVP — M0+M1.

Faithful to paper https://arxiv.org/html/2610.31329 §2 stages:
  1) concept discovery + knowledge graph
  2) multi-expert localization (SAM3 + in-house DINO-style OVD)
  3) VLM verification (Qwen3-VL)
  4) SAM2 mask
  5) matting
  + feedback / revision cross-cut

This MVP uses honest mocks for heavy models; interfaces are pluggable.
"""

__version__ = "0.1.0"
