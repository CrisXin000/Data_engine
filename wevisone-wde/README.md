# WeVisOne-WDE MVP（M0 + M1 + M2 指代 + M3 mask + M4 alpha）

复现 [WeVisOne-VL](https://arxiv.org/html/2610.31329) 论文 **§2 Data Engine (WDE)** 的可运行最小闭环：

**读图 → 生成 query → 多专家候选（mock / 真实开源替代）→ 核验（rule / CLIP 语义，可选 Florence 尝试）→ 已核验实例二值 mask（可选）→ 连续 alpha matting（可选）→ 写表 → 导出**

> 本仓库是**工程复现骨架**，不是官方实现。论文重模型（SAM3、内部 DINO 风格 OVD、Qwen3-VL、SAM2、BiRefNet/MatAnyone2）在 MVP 中为 **诚实 mock / 开源替代 / 接口预留**，不会假装已接入 SAM3 或内部检测器。

## 论文阶段对照

| 阶段 | 论文 §2 | 本 MVP |
|------|---------|--------|
| ① 语义发现与图谱 | Tokenize Anything + Qwen3-VL 指代；WordNet/Wikipedia KG | mock / YOLO-World 打标；**指代**：`structured`/`grounded`（检测框+crop 颜色/位置，开源站位 Qwen3-VL）或 mock 模板；`qwen3_vl` 配置名仅为别名→structured；NLTK WordNet 种子 |
| ② 多专家定位 | **SAM3** + **内部 DINO 风格开放词汇检测器** | **mock**，或 **开源替代**：`YoloWorldExpert`（站位 Expert A）+ `Owlv2Expert`（站位 Expert B） |
| ③ VLM 核验 | Qwen3-VL | `rule`（IoU+score）；`vlm_json`/`clip_semantic`（**CLIP ViT-B/32** 弱语义替代，诚实标注 `stands_in_for: Qwen3-VL`；Florence-2 尝试失败则自动回退） |
| ④ SAM2 mask | SAM2 | **可选**：`ultralytics` **SAM2-Tiny**（`sam2_t.pt`）做 box-prompt 分割；失败则 MobileSAM，再失败则 `box_fill`（框内全 1）。**不是**官方 Meta `sam2` 包 / SAM2 全家桶。默认仍跳过 |
| ⑤ matting | BiRefNet / MatAnyone2 等 | **可选**：ONNX **BiRefNet-General-Lite**（Swin-Tiny）在实例 crop 上出连续 α，再按 IoU/coverage/area ratio 相对二值 mask S 筛选；不通过则羽化/复制 S。**不是** MatAnyone2，也**没有**训练论文里的专用 matting annotator |
| 回流 | 错误聚类与 revision | `error_record` / `revision` 表骨架 |

**注意（论文事实）**：§2 检测专家写的是 SAM3 + 内部 DINO 风格 OVD；**没有写 Rex-Omni**。Rex-Omni 出现在模型/评测相关章节，不要当成 WDE 专家事实。

## 工程推断（非论文原文，已标明）

- SQLite + SQLAlchemy；坐标统一 **xyxy 原图像素**
- 表带 `created_at` / `run_id` / `data_version` / `valid_status`；专家原始输出 append-only
- **真实专家是开源替代，不是论文模型**：YOLO-World ≈ Expert A（SAM3），OWLv2 ≈ Expert B（内部 DINO OVD）
- 指代生成（M2）：`structured` 用 YOLO-World 框 + crop 主色/象限/面积排序组装英文指代与 `constraints`（**不是** Qwen3-VL）；专家提议仍用 `extract_class_names` 抽核心名词
- 合成几何黄金集仅适合 mock 回归；真实检测器用 `data/gold/images_real/`

## 已实现范围

- **M0**：合成黄金集（5 张）+ schema + KG + CLI
- **M1**：类别 box 单线（discover → multi-expert propose → verify → instance_link → export）
- **M2 指代**：`StructuredReferringGenerator`（配置名 `structured` / `grounded`）— 看检测结果接地，写出 expression + constraints；保留 mock；`qwen3_vl` 为别名（可选轻量 caption，失败回落 structured，meta 记 fallback）
- **Real detectors（CPU）**：`YoloWorldExpert` + `Owlv2Expert`（transformers 不可用时退回第二套 YOLO-World 配置，并在 README / raw_output 标明弱双专家）
- **语义核验（CPU）**：`clip_semantic` / `vlm_json`（CLIP crop↔query；**不是** Qwen3-VL）。流水线在专家 propose 结束后释放检测权重再加载核验器
- **M3 二值 mask**：`segment_instance`；`--segment sam2` 启用。优先 Ultralytics SAM2-Tiny（站位论文 SAM2，`stands_in_for: SAM2`，`mock: false`）；不行再 MobileSAM / `box_fill`（会标明弱于 SAM2）。stub 保留，默认 `skip_segment: true`
- **M4 连续 alpha**：`matte_instance`；`--matte birefnet_lite` 启用。优先 ONNX BiRefNet-General-Lite（站位论文 BiRefNet，`stands_in_for: BiRefNet/MatAnyone2`，`mock: false`）；门限失败或模型失败则羽化 S（`alpha_source=feather_s`）并写 `fallback_reason`。stub 保留，默认 `skip_matte: true`

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
| `discover.referring_generator` | `mock` | **`structured`**（检测框接地；也可用 `grounded` / `qwen3_vl` 别名；**不是**真 Qwen3-VL） |

`raw_output` 中 `mock: false`，并写明 `model` / `stands_in_for`（论文专家替代：YOLO-World / OWLv2）。

### CPU 预期

- 首跑会下载 YOLO-World 权重（~25MB）+ CLIP 文本塔 + OWLv2（数百 MB）
- 单张 720p 量级：YOLO-World 约数秒；OWLv2 约数秒～十几秒（会缩到 max_side=768）
- **无 nvidia-smi / 无 GPU 属预期**


## M2：接地指代 query 生成（开源站位 Qwen3-VL）

论文用 **Qwen3-VL** 生成组合式指代表达（属性/状态/动作/关系）。本仓库**没有**接入 Qwen3-VL。

| 配置名 | 含义 |
|--------|------|
| `mock` | 死模板（`the {color} {shape} on the left` 等），仅回归 |
| `structured` / `grounded` | **主实现**：YOLO-World 检框 → crop 估计主色/相对位置/面积排序 → 英文指代 + `meta.constraints`（attributes/states/actions/relations/part_scope）；未知字段显式空/`unknown`，不编造不可见图属性 |
| `qwen3_vl` | **别名**：优先尝试可选轻量 caption（BLIP，默认不强制）；失败/OOM → 回落 `structured`，`meta.fallback` 记录。**仍然不是 Qwen3-VL** |

每图至少产出：若干正例指代（单目标；多实例可有 largest / leftmost / near X）+ **≥1 条 absent/负条件**（`is_negative` / `meta.is_negative_hint` / `constraints.absent=true`）。

`Query` 表无独立 `constraints_json` 列；约束写入 **`meta.constraints`**（另含 `mock:false`、`model`/`backend`、`stands_in_for:"Qwen3-VL"`、`base_category`、box/颜色直方图等 evidence）。

### 一条命令

```bash
source .venv/bin/activate
bash scripts/smoke_referring.sh
```

手动：

```bash
python -m wde.cli run --images data/gold/images_real --mode real --referring structured --verify rule
python -m wde.cli export --run-id <RUN_ID> --out data/runs/<RUN_ID>/export.json
python scripts/export_viz.py --export data/runs/<RUN_ID>/export.json
python scripts/write_referring_report.py --export data/runs/<RUN_ID>/export.json --out-dir data/runs/<RUN_ID>
```

对比报告默认写到 `data/runs/compare_mock_vs_structured_referring/`（`REFERRING.md` + `referring_queries.json`）。

### 指代 + 核验的已知弱项

- 专家对指代仍用名词启发式（`extract_class_names`）；CLIP/`vlm_json` 核验会**降级到核心名词**再比相似度（`raw_output.prompt_note`），**整句指代语义核验仍偏弱**
- 颜色来自 crop 直方图最近邻，暗色车辆等可能标成 `black`/`gray`，保守但不保证语义「正确」
- 关系词仅在存在**第二个可见异类参照**时生成（如 `near the bus`）
- 可选 BLIP caption 默认关闭；内存紧时跳过。**不要**默认拉 Qwen2-VL / Qwen3-VL

## 启用 VLM / 语义核验（替代 rule）

论文核验器是 **Qwen3-VL**。本仓库**没有**接入 Qwen3-VL；工程替代如下：

| backend | 含义 |
|---------|------|
| `rule` | IoU 去重 + detector score（默认；mock 模式强制） |
| `clip_semantic` / `clip` | OpenAI CLIP ViT-B/32：整图上下文 + padded crop 与 query 的余弦相似度；再 IoU 去重 |
| `vlm_json` / `vlm` | 先尝试 `microsoft/Florence-2-base`；当前 **transformers 5.x** 下 Florence remote code 加载失败时**自动回退 CLIP**，并在 `raw_output.fallback_reason` 记录原因 |
| `florence2` | 直接 Florence（无自动回退；当前环境通常会报错） |

### 一条命令（推荐）

```bash
source .venv/bin/activate
# 需已有真实检测器依赖 + weights/clip/ViT-B-32.pt（见 setup_real_detectors / 已缓存权重）
bash scripts/smoke_vlm_verify.sh
```

手动：

```bash
python -m wde.cli run --images data/gold/images_real --mode real --verify vlm_json
# 或只开 CLIP：
python -m wde.cli run --images data/gold/images_real --mode real --verify clip_semantic
python -m wde.cli export --run-id <RUN_ID> --out data/runs/<RUN_ID>/export.json
python scripts/export_viz.py --export data/runs/<RUN_ID>/export.json
```

也可改 `config/default.yaml` → `models.verify.backend: vlm_json`（`--mode mock` 仍强制 `rule`，避免拖慢 mock 回归）。

### 决策语义

- `accept` / `reject` / `uncertain` / `duplicate`
- CLIP：`sim_crop >= clip_accept_min` → accept；`sim_crop < clip_reject_max` → reject；中间带 → **uncertain**（**不会把 unknown 当 accept**）
- 同类高 IoU 仍做后处理去重（`pipeline.iou_dedup_threshold`）
- 类别 query 为主路径；指代 query 降级为抽核心名词再比对（`raw_output.prompt_note` 标明）
- 每条 `raw_output` 含：`mock: false`、真实模型名、`stands_in_for: "Qwen3-VL"`、`prompt` / `prompt_version`、相似度摘要

### CPU 内存与耗时（经验）

- 环境约 15Gi RAM；**不要**与 YOLO-World/OWLv2 同时常驻——runner 会在 propose 后 `release_expert_models` 再懒加载 CLIP
- CLIP ViT-B/32 权重约 350MB（`weights/clip/ViT-B-32.pt`）；加载后峰值通常 <1Gi 增量
- 每候选一次 crop 编码：CPU 上约 0.1–0.5s/框；2 张图整次核验大约数十秒量级（另加检测时间）
- Florence-2-base：当前 transformers 5.22 加载失败（`Florence2LanguageConfig.forced_bos_token_id`）；不要假装已接入
- 失败策略：`vlm_json` 默认 Florence→CLIP；若 CLIP 也失败则抛错。设 `models.verify.fallback_to_rule: true` 可再退回 rule（`raw` 会记录 fallback）

### 与 rule 对比

`bash scripts/smoke_vlm_verify.sh` 会写 `data/runs/compare_rule_vs_vlm/COMPARISON.md`。
典型差异：rule 容易接受「分数够但语义不对」的框；CLIP 会按 query 文本拒掉明显不匹配的 crop，并可能给出 `uncertain`。


## M3：已核验实例的二值 mask（开源站位 SAM2）

论文用 **SAM2** 给已核验框出二值 mask。本仓库**没有**接入官方 Meta `sam2` 包，也没有拉 SAM2 大权重全家桶。

| backend | 含义 |
|---------|------|
| `stub` | 不写文件（默认；`skip_segment: true`）。接口保留 |
| `sam2` | **主路径**：`ultralytics.SAM("sam2_t.pt")`，即 **SAM2-Tiny** 做 box-prompt。`meta.model` 写真实权重路径，`stands_in_for: "SAM2"`，`mock: false`。失败 → 尝试 MobileSAM；再失败 → `box_fill`，并在 `meta.note` / `weaker_than_sam2` 写明 |
| `mobile_sam` | 直接用 Ultralytics MobileSAM（提示分割，**弱于 SAM2**） |
| `box_fill` | 弱回退：把 prompt 框内像素全部写成 255。**不是**学习式分割，明确弱于 SAM2，不会假装成 SAM2 |

默认 mock / 未传 `--segment` 时仍然 **skip**。流水线在 propose → verify 之后、卸载检测器与 VLM/CLIP 之后才加载分割模型。

输出：与原图同尺寸的 PNG（`L` 模式，前景=255），路径形如 `data/runs/<run_id>/masks/<instance_uid>.png`；`mask_alpha` 行通过 `instance_link_id` 以及 `meta.instance_uid` / `query_id` / `query_text` 继承该实例的 query 关联。

### 一条命令

```bash
source .venv/bin/activate
bash scripts/smoke_segment.sh
```

手动：

```bash
python -m wde.cli run --images data/gold/images_real --mode real --verify rule --segment sam2
python -m wde.cli export --run-id <RUN_ID> --out data/runs/<RUN_ID>/export.json
python scripts/export_viz.py --export data/runs/<RUN_ID>/export.json
# 框：viz/*_boxes.png；mask 叠加：viz/*_masks.png
```

配置见 `config/default.yaml`：

- `models.segment.backend`: `stub` \| `sam2` \| `mobile_sam` \| `box_fill`
- `models.segment.weights`: `weights/segment/sam2_t.pt`
- `models.segment.device`: `cpu`
- `models.segment.imgsz`: `512`（官方常用 1024；512 是为了 CPU 省内存）
- `pipeline.skip_segment`: 默认 `true`；CLI `--segment sam2` 会改成 `false`

### CPU 预期与限制

- 环境约 15Gi RAM，可用内存经常只有 2–3Gi；**不要**与 YOLO-World / OWLv2 / CLIP 同时常驻
- SAM2-Tiny 权重约 **75MB**（ultralytics `sam2_t.pt`）；`imgsz=512` 时加载+推理增量大约数百 MB，单框 CPU 约 0.3–2s（视图大小和负载）
- 这是 **Ultralytics 封装的 SAM2-Tiny**，不是 Meta 官方仓库、不是 `sam2_s/b/l`、不做视频记忆
- MobileSAM / `box_fill` 只在 Tiny 失败或你显式指定时使用，并会写明弱于 SAM2
- 二值 mask 之后可接 matting（见下一节）

### 黄金集说明

| 目录 | 用途 |
|------|------|
| `data/gold/images/` | 合成几何图（红矩形等）— **仅 mock 回归**；YOLO-World 对「red rectangle」几乎无效 |
| `data/gold/images_real/` | 真实样例（ultralytics `bus.jpg` / `zidane.jpg`，Wikimedia dog）+ `*.labels.json` |


## M4：连续 alpha matting（开源站位 BiRefNet / MatAnyone2）

论文 §2.6：对实例二值 mask **S**，在 crop 上分别估计 αᴮ（**BiRefNet**）与 αᴹ（**MatAnyone2**），按门限筛选；失败则回退到 S。之后他们还会训练专用 matting annotator 全库重标。

本仓库**没有**接入 MatAnyone2（无视频式记忆/传播），**没有**训练论文 annotator，也**没有**跑完整 BiRefNet-matting 大权重。CPU MVP 只实现简化版：

| backend | 含义 |
|---------|------|
| `stub` | 不写文件（默认；`skip_matte: true`）。接口保留 |
| `birefnet_lite` / `birefnet` | **主路径**：ONNX [BiRefNet-General-Lite](https://github.com/ZhengPeng7/BiRefNet)（`BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx`，Swin-Tiny，约 214MB）。在 box/S 外扩 crop 上推理，把连续 α（sigmoid，0–255 PNG）贴回原图尺寸。`meta.model` 写真实权重路径，`stands_in_for: "BiRefNet/MatAnyone2"`，`mock: false` |
| `birefnet_matanyone` | **别名** → `birefnet_lite`。名字保留是为了对照论文双模型设定；**不会**跑 MatAnyone2 |
| `feather` | 不跑网络：对 S 做距离变换斜坡（有 scipy）或高斯羽化 |
| `copy_s` | 不跑网络：原样复制 S（仍然是接近二值的 alpha） |

筛选（可配置子集，默认是工程取值，**不是**从论文抄来的原数字）：

- `IoU(binarize(α), S) >= iou_min`（默认 0.5）
- `coverage = |α∩S| / |S| >= coverage_min`（默认 0.6）
- `area_ratio = |α| / |S| ∈ [area_ratio_min, area_ratio_max]`（默认 0.5–2.0）

不满足或 ONNX 加载/推理失败：回退 `feather`（或配置的 `copy_s`），并写 `alpha_source` / `fallback_reason` / `fallback: true`。流水线在 segment 之后会 `release_segment_model()` 再加载 matting，避免和 SAM2 同时占内存。

输出：与原图同尺寸的 PNG（`L` 模式，0–255），路径形如 `data/runs/<run_id>/alphas/<instance_uid>.png`。

### 一条命令

```bash
source .venv/bin/activate
# 需要 SAM2-Tiny 权重 + onnxruntime（pip install 'onnxruntime>=1.17' 或 pip install -e '.[matte]'）
bash scripts/smoke_matte.sh
```

手动：

```bash
python -m wde.cli run --images data/gold/images_real --mode real --verify rule --segment sam2 --matte birefnet_lite
python -m wde.cli export --run-id <RUN_ID> --out data/runs/<RUN_ID>/export.json
python scripts/export_viz.py --export data/runs/<RUN_ID>/export.json
# 框：viz/*_boxes.png；二值 mask：viz/*_masks.png
# alpha 棋盘格合成：viz/*_alphas.png；S vs α 裁剪对照：viz/*_alpha_vs_mask.png
```

配置见 `config/default.yaml`：

- `models.matte.backend`: `stub` \| `birefnet_lite` \| `feather` \| `copy_s`
- `models.matte.weights`: `weights/matte/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx`
- `models.matte.device`: `cpu`
- `models.matte.imgsz`: `1024`（该 ONNX 按 1024×1024 导出）
- `models.matte.iou_min` / `coverage_min` / `area_ratio_min` / `area_ratio_max`
- `pipeline.skip_matte`: 默认 `true`；CLI `--matte birefnet_lite` 会改成 `false`

### CPU 预期与限制

- 环境约 15Gi RAM；**不要**与检测器 / CLIP / SAM2 同时常驻——runner 会在 segment 后卸载分割模型再加载 ONNX
- Lite ONNX 约 **214MB**；1024 输入在 CPU 上单实例大约数秒到十几秒（视 crop 与负载）
- 这是 **BiRefNet-General-Lite（显著前景）**，不是 BiRefNet-matting 专用头，也不是实例条件模型：靠 crop 把当前实例当成「前景」
- **未接 MatAnyone2**，没有双 matte 互证，没有视频传播
- **未训练**论文中的专用 matting annotator，也不会全库重标
- 门限过严时会大量回退到羽化 S，边缘比真 matting 粗，但至少是连续 α，并在 meta 里诚实标注

## CLI

| 命令 | 作用 |
|------|------|
| `python -m wde.cli init-db` | 建表 |
| `python -m wde.cli kg init` | WordNet/fallback 种子概念 |
| `python -m wde.cli kg add <term>` | 候选词 → 文本链接 |
| `python -m wde.cli ingest-gold` | 生成并入库合成黄金集 |
| `python -m wde.cli run --images ... --mode mock\|real [--referring mock\|structured\|grounded\|qwen3_vl] [--verify rule\|vlm_json\|clip_semantic] [--segment stub\|sam2\|mobile_sam\|box_fill\|skip] [--matte stub\|birefnet_lite\|feather\|skip]` | 端到端流水线 |
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
    smoke_vlm_verify.sh
    smoke_referring.sh
    smoke_segment.sh
    smoke_matte.sh
    write_vlm_comparison.py
    write_referring_report.py
    export_viz.py
  src/wde/
    propose/experts.py   # mock + YoloWorldExpert + Owlv2Expert
    discover/tagger.py   # mock + YOLO-World category tagger; build_referring_generator
    discover/referring.py  # structured/grounded referring (Qwen3-VL stand-in)
    ...
```

## 哪些是 mock / 哪些是真实替代

| 组件 | 状态 |
|------|------|
| 类别发现 | mock（sidecar）；real 模式 → YOLO-World 或 sidecar |
| 指代生成 | mock 模板 **或** `structured`/`grounded`（接地开源站位）；`qwen3_vl`=别名→structured（**不是**真 Qwen3-VL） |
| Expert A | mock **或** `YoloWorldExpert`（**论文 SAM3 的开源替代**） |
| Expert B | mock **或** `Owlv2Expert`（**论文内部 DINO OVD 的开源替代**） |
| 核验 | `rule`；`vlm_json`/`clip_semantic`（**CLIP 弱语义**，站位 Qwen3-VL；非 Qwen3-VL） |
| SAM2 mask | stub（默认跳过）**或** Ultralytics SAM2-Tiny（站位论文 SAM2；非官方 Meta sam2）→ MobileSAM → `box_fill` |
| matting | stub（默认跳过）**或** ONNX BiRefNet-General-Lite（站位论文 BiRefNet/MatAnyone2；**不是** MatAnyone2，也未训论文 annotator）→ 门限失败则羽化/复制 S |

## 已知限制

- 合成几何图不适用真实开放词汇检测；请用 `images_real/`
- **没有**接入 SAM3 / 内部 DINO / Qwen3-VL；双专家是 YOLO-World + OWLv2；指代是 structured 接地生成；核验是 CLIP（或失败的 Florence 尝试）——均为开源替代，弱于论文设定
- 指代核验仍偏弱：CLIP/专家对指代句降级抽名词，非整句组合式语义
- OWLv2 不可用时自动退回「同 YOLO-World 不同 conf」——属弱双专家，非论文设定
- Florence-2 在 transformers 5.x 下当前无法加载；`vlm_json` 回退 CLIP 并写明原因
- CPU 推理较慢、内存紧：先 propose 再释放检测器，再核验，再释放 CLIP，再分割并卸载 SAM，最后才加载 matting ONNX；默认仍跳过 segment / matte
- matting 是 BiRefNet-General-Lite 单模型 + 相对 S 的门限，不是论文的 BiRefNet+MatAnyone2 双估计，也没有专用 annotator
- 本环境可能无法下载 NLTK WordNet；`kg init` 会回退内置概念

## 许可

代码 Apache-2.0。论文与 WeVisOne-VL 权重归属原作者（Tencent WeChat Vision）。合成黄金集为本地生成；`images_real/` 来自 Ultralytics 样例与 Wikimedia，仅供演示。
