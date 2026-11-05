# WeVisOne-WDE MVP（M0 + M1）

复现 [WeVisOne-VL](https://arxiv.org/html/2610.31329) 论文 **§2 Data Engine (WDE)** 的可运行最小闭环：

**读图 → 生成 query → 多专家候选（mock / 真实开源替代）→ VLM 核验占位（规则）→ 写表 → 导出**

> 本仓库是**工程复现骨架**，不是官方实现。论文重模型（SAM3、内部 DINO 风格 OVD、Qwen3-VL、SAM2、BiRefNet/MatAnyone2）在 MVP 中为 **诚实 mock / 开源替代 / 接口预留**，不会假装已接入 SAM3 或内部检测器。

## 论文阶段对照

| 阶段 | 论文 §2 | 本 MVP |
|------|---------|--------|
| ① 语义发现与图谱 | Tokenize Anything + Qwen3-VL 指代；WordNet/Wikipedia KG | mock 标签器 / 可选 YOLO-World 打标；模板指代；NLTK WordNet 种子 |
| ② 多专家定位 | **SAM3** + **内部 DINO 风格开放词汇检测器** | **mock**，或 **开源替代**：`YoloWorldExpert`（站位 Expert A）+ `Owlv2Expert`（站位 Expert B） |
| ③ VLM 核验 | Qwen3-VL | 规则核验（IoU 去重 + score）；`vlm_json` 接口预留 |
| ④ SAM2 mask | SAM2 | stub（M1 默认跳过） |
| ⑤ matting | BiRefNet / MatAnyone2 等 | stub（M1 默认跳过） |
| 回流 | 错误聚类与 revision | `error_record` / `revision` 表骨架 |

**注意（论文事实）**：§2 检测专家写的是 SAM3 + 内部 DINO 风格 OVD；**没有写 Rex-Omni**。Rex-Omni 出现在模型/评测相关章节，不要当成 WDE 专家事实。

## 工程推断（非论文原文，已标明）

- SQLite + SQLAlchemy；坐标统一 **xyxy 原图像素**
- 表带 `created_at` / `run_id` / `data_version` / `valid_status`；专家原始输出 append-only
- **真实专家是开源替代，不是论文模型**：YOLO-World ≈ Expert A（SAM3），OWLv2 ≈ Expert B（内部 DINO OVD）
- 指代 query 用简单启发式抽核心名词再 `set_classes` / text queries（`extract_class_names`）
- 合成几何黄金集仅适合 mock 回归；真实检测器用 `data/gold/images_real/`

## 已实现范围

- **M0**：合成黄金集（5 张）+ schema + KG + CLI
- **M1**：类别 box 单线（discover → multi-expert propose → rule verify → instance_link → export）
- **Real detectors（CPU）**：`YoloWorldExpert` + `Owlv2Expert`（transformers 不可用时退回第二套 YOLO-World 配置，并在 README / raw_output 标明弱双专家）
- segment / matte：接口占位，默认跳过

## 环境

- Python **3.11+**（已在 3.13 验证）
- **无 GPU 也可跑**；默认 mock 全程 CPU；real 模式用 CPU 版 torch（较慢）

## 一条命令复现（mock，推荐）

```bash
cd /workspace/wevisone-wde
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python -m wde.cli init-db \
  && python -m wde.cli kg init \
  && python -m wde.cli ingest-gold \
  && python -m wde.cli run --images data/gold/images --mode mock \
  && python -m wde.cli status
```

或：

```bash
bash scripts/smoke_m1.sh
```

## 切换到真实检测器（CPU）

```bash
source .venv/bin/activate
bash scripts/setup_real_detectors.sh   # CPU torch + ultralytics + transformers + 权重
bash scripts/smoke_real_detectors.sh   # 在 data/gold/images_real/ 上跑 1–2 张
```

手动跑：

```bash
python -m wde.cli run --images data/gold/images_real --mode real
# 记下 run_id 后：
python -m wde.cli export --run-id <RUN_ID> --out data/runs/<RUN_ID>/export.json
python scripts/export_viz.py --export data/runs/<RUN_ID>/export.json
```

配置（`config/default.yaml` 或 CLI）：

| 键 | mock | real（`--mode real` 默认） |
|----|------|---------------------------|
| `models.mode` | `mock` | `real` |
| `propose.expert_a` | `mock` | `yolo_world`（`yolov8s-worldv2.pt`） |
| `propose.expert_b` | `mock` | `owlv2`（`google/owlv2-base-patch16`） |
| `discover.category_tagger` | `mock` | `yolo_world`（有 `*.labels.json` 时优先读 sidecar） |

`raw_output` 中 `mock: false`，并写明 `model` / `stands_in_for`（论文专家替代：YOLO-World / OWLv2）。

### CPU 预期

- 首跑会下载 YOLO-World 权重（~25MB）+ CLIP 文本塔 + OWLv2（数百 MB）
- 单张 720p 量级：YOLO-World 约数秒；OWLv2 约数秒～十几秒（会缩到 max_side=768）
- **无 nvidia-smi / 无 GPU 属预期**

### 黄金集说明

| 目录 | 用途 |
|------|------|
| `data/gold/images/` | 合成几何图（红矩形等）— **仅 mock 回归**；YOLO-World 对「red rectangle」几乎无效 |
| `data/gold/images_real/` | 真实样例（ultralytics `bus.jpg` / `zidane.jpg`，Wikimedia dog）+ `*.labels.json` |

## CLI

| 命令 | 作用 |
|------|------|
| `python -m wde.cli init-db` | 建表 |
| `python -m wde.cli kg init` | WordNet/fallback 种子概念 |
| `python -m wde.cli kg add <term>` | 候选词 → 文本链接 |
| `python -m wde.cli ingest-gold` | 生成并入库合成黄金集 |
| `python -m wde.cli run --images ... --mode mock\|real` | 端到端流水线 |
| `python -m wde.cli export --run-id <id> --out ...` | 导出 JSON |
| `python -m wde.cli status` | 表行数与最近 run |

## 目录

```
wevisone-wde/
  config/default.yaml
  data/{images,gold,runs,models}/
  scripts/
    smoke_m1.sh
    setup_real_detectors.sh
    smoke_real_detectors.sh
    export_viz.py
  src/wde/
    propose/experts.py   # mock + YoloWorldExpert + Owlv2Expert
    discover/tagger.py   # mock + YOLO-World category tagger
    ...
```

## 哪些是 mock / 哪些是真实替代

| 组件 | 状态 |
|------|------|
| 类别发现 | mock（sidecar）；real 模式 → YOLO-World 或 sidecar |
| 指代生成 | **mock 模板**；`qwen3_vl` 预留 |
| Expert A | mock **或** `YoloWorldExpert`（**论文 SAM3 的开源替代**） |
| Expert B | mock **或** `Owlv2Expert`（**论文内部 DINO OVD 的开源替代**） |
| 核验 | rule；`vlm_json` 预留 |
| SAM2 / matting | stub，默认跳过 |

## 已知限制

- 合成几何图不适用真实开放词汇检测；请用 `images_real/`
- **没有**接入 SAM3 / 内部 DINO / Qwen3-VL；双专家是 YOLO-World + OWLv2 开源替代（弱于论文设定）
- OWLv2 不可用时自动退回「同 YOLO-World 不同 conf」——属弱双专家，非论文设定
- CPU 推理较慢；segment / matte 默认跳过
- 本环境可能无法下载 NLTK WordNet；`kg init` 会回退内置概念

## 许可

代码 Apache-2.0。论文与 WeVisOne-VL 权重归属原作者（Tencent WeChat Vision）。合成黄金集为本地生成；`images_real/` 来自 Ultralytics 样例与 Wikimedia，仅供演示。
