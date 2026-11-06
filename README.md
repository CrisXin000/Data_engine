# Data_engine

WeVisOne-VL Data Engine (WDE) 的工程复现：可运行 MVP（M0–M4）。

- 实现与说明见 [`wevisone-wde/`](./wevisone-wde/)
- 阶段对照、开源替代与用法见 [`wevisone-wde/README.md`](./wevisone-wde/README.md)
- 论文：https://arxiv.org/html/2610.31329

快速开始：

```bash
cd wevisone-wde
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
bash scripts/smoke_m1.sh          # mock 闭环
# 真实检测器（CPU）：
bash scripts/setup_real_detectors.sh
bash scripts/smoke_real_detectors.sh
# 可选后续阶段：
# bash scripts/smoke_referring.sh    # 接地指代
# bash scripts/smoke_vlm_verify.sh   # CLIP 语义核验
# bash scripts/smoke_segment.sh      # SAM2-Tiny 二值 mask
# bash scripts/smoke_matte.sh        # BiRefNet-lite 连续 alpha
```

权重（YOLO-World / OWLv2 / CLIP / SAM2-Tiny `.pt` / BiRefNet-General-Lite `.onnx`）不纳入本仓库。它们是论文模型的开源站位：YOLO-World / OWLv2 ≠ SAM3 / 内部 DINO，CLIP ≠ Qwen3-VL，Ultralytics SAM2-Tiny ≠ 官方 SAM2，BiRefNet-General-Lite ≠ MatAnyone2。
