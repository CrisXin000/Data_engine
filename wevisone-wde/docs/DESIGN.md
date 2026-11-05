# WeVisOne-VL 数据引擎 · 可复现设计方案（摘要）

完整可交互版：https://cursor.com/canvas/store-693791af-e96f-4b77-b641-495ebe99e556/70224b72-facd-43fc-bca3-60122343d1a5

论文：https://arxiv.org/html/2610.31329 （§2 Data Engine）

## 阶段

1. 语义发现与知识图谱（类别 + 指代 query）
2. 多专家定位（论文：SAM3 + 内部 DINO 风格 OVD；本仓库开源替代：YOLO-World + OWLv2）
3. VLM 语义核验（论文：Qwen3-VL；本仓库 MVP：规则 IoU）
4. SAM2 二值 mask（接口预留）
5. 连续 alpha matting（接口预留）
6. 难例回流与版本发布（表骨架）

## 工程约定

- 原始候选 / 核验 / 专家输出 append-only，禁止原地覆盖
- 坐标统一 xyxy 原图像素
- 表字段含 `created_at` / `run_id` / `data_version` / `valid_status`
- 论文事实与工程推断在 README / 代码注释中分开标注
