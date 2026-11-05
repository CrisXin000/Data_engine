# Data_engine

WeVisOne-VL Data Engine (WDE) 的工程复现：设计文档 + 可运行 MVP。

- 实现与说明见 [`wevisone-wde/`](./wevisone-wde/)
- 设计方案摘要见 [`wevisone-wde/docs/DESIGN.md`](./wevisone-wde/docs/DESIGN.md)
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
```

权重（YOLO-World / OWLv2）首次运行时自动下载，不纳入本仓库。
