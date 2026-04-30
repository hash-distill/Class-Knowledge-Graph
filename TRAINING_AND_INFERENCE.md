# 训练与预测运行手册

本文档集中维护本项目全部训练、评估、推理命令。
建议按"13 类标准流程"先跑通一轮，再进行对比实验。

---

## 1. 环境准备

```bash
cd ..

# 可按需修改环境名
conda create -n classkg python=3.12 -y
conda activate classkg

pip install -U pip
pip install -r requirements.txt

# 若提示缺少 ultralytics
pip install -U ultralytics
```

---

## 2. 数据准备与校验

### 2.1 构建 SCB-5 稳健 3 类数据集

```bash
cd Class_Detection
python tools/build_scb5_unified.py
```

说明：
- 输入目录为 `../SCB-Dataset/SCB-Dataset`（包含 9 个 SCB-5 原始子集）。
- 输出目录为 `../SCB5_yolo_unified`。
- 脚本会为每个子集独立重映射局部 class ID 到全局 3 类 ID `0..2`。
- 自动去除重复图片（合并多子集中相同图片的标注）。
- 自动清洗越界坐标到 `[0, 1]`，并丢弃无效框。

### 2.2 数据完整性审计

```bash
cd Class_Detection

python tools/dataset_audit.py \
  --dataset-root ../SCB5_yolo_unified \
  --output artifacts/reports/scb5_audit.json
```

### 2.3 快速检查类别分布与坐标越界

```bash
cd ..

# 类别分布（3 类应为 0..2）
awk '{c[$1]++} END {for (k in c) print k, c[k]}' SCB5_yolo_unified/labels/train/*.txt | sort -n
awk '{c[$1]++} END {for (k in c) print k, c[k]}' SCB5_yolo_unified/labels/val/*.txt | sort -n

# 越界检查（正常应无输出）
find SCB5_yolo_unified -name "*.txt" -exec \
awk '$2 < 0 || $2 > 1 || $3 < 0 || $3 > 1 || $4 < 0 || $4 > 1 || $5 < 0 || $5 > 1 {print FILENAME, $0}' {} +
```

---

## 3. 检测模型训练

以下命令均在 `Class_Detection` 目录执行。

### 3.1 SCB-Dataset 7 类行为检测训练（当前主干模型）

使用全新的 SCB-Dataset 数据集直接训练学生的 7 类动作（读、写、抬头、举手、转头、站立、讨论）。

```bash
cd Class_Detection

python scripts/train_det.py \
  --data configs/scb_dataset_yolo.yaml \
  --model yolo26m.pt \
  --epochs 60 \
  --imgsz 960 \
  --batch 32 \
  --device 3 \
  --workers 8 \
  --patience 10 \
  --name scbehavior_yolo26m \
  --cache
```

### 3.2 兜底环境模型（Dual-Model 兜底策略）

SCBehavior 存在大量“漏标安静听课学生”的缺陷。为了拯救这些被漏标的背景学生，**不需要额外训练**。系统直接引入 Ultralytics 官方预训练的 **`yolo26m.pt`** (COCO 80类) 作为兜底的 `env_model`，在推理阶段通过 NMS 自动找回所有的学生（统一赋予 `attending` 状态），并提取黑板作为 OCR 锚点。

---

## 4. 可选训练

### 4.1 YOLO Pose 微调（可选）

```bash
cd Class_Detection

python scripts/train_pose.py \
  --data <pose_dataset_yaml> \
  --model yolo26n-pose.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 32 \
  --device 0 \
  --name classroom_pose
```

### 4.2 ST-GCN 动作分类训练（可选）

要求 `--keypoints-dir` 下已有 `train/`、`val/` 与 `label.json`。

```bash
cd Class_Detection

python scripts/train_stgcn.py \
  --config configs/stgcn.yaml \
  --keypoints-dir <keypoints_dataset_dir> \
  --epochs 100 \
  --device 0
```

---

## 5. 评估命令

### 5.1 检测评估

```bash
cd Class_Detection

python scripts/eval_det.py \
  --weights yolo26m.pt \
  --data configs/scb_yolo.yaml \
  --imgsz 960 \
  --batch 32 \
  --device 0


python scripts/eval_det.py \
  --weights artifacts/runs/detect/scb5_yolo26m_e40/weights/best.pt \
  --data configs/scb_yolo.yaml \
  --imgsz 960 \
  --batch 16 \
  --device 0
```

提示：评估和部署优先使用 `best.pt`，不要默认使用 `last.pt`。

---

## 6. 预测与推理命令

### 6.1 视频推理（端到端）

#### 使用双模型架构推理（推荐：7 类行为 + COCO 兜底）

```bash
cd Class_Detection

python scripts/infer_video.py \
  --source ../classroom.mp4 \
  --config configs/pipeline.yaml \
  --det-weights artifacts/runs/detect/scbehavior_yolo26m/weights/best.pt \
  --env-weights yolo26m.pt \
  --pose-weights yolo26n-pose.pt \
  --device 0 \
  --interval-sec 1.0 \
  --save \
  --output artifacts/results/latest
```

> **说明**：此模式使用了最新的双模型架构。`--det-weights` 指定了预测学生 7 种行为的模型，而 `--env-weights yolo26m.pt` 则引入了 COCO 大模型来兜底找回所有“安静听课”的学生。同时，系统默认加载了 `configs/bytetrack_low.yaml` 追踪器。若发现追踪丢框，可尝试将 `--interval-sec 1.0` 调整为 `0`（逐帧预测）。

#### 使用预训练 yolo26m.pt 直接推理（降级体验版，无需训练）

官方预训练权重（COCO 80类）无法自动识别"黑板"或"屏幕"，因此**默认不会触发 OCR**。为了在免训练的情况下体验完整的知识点提取流程，你必须**手动指定 PPT/屏幕 的区域坐标** (`--ppt-crop`)，强制开启区域 OCR。

**Linux / macOS (Bash)**
```bash
cd Class_Detection

python scripts/infer_video.py \
  --source ../classroom.mp4 \
  --config configs/pipeline.yaml \
  --device 0 \
  --interval-sec 1.0 \
  --ppt-crop "100,50,800,600" \
  --save \
  --output artifacts/results/latest
```

**Windows (PowerShell)**
```powershell
# 确保在 Class_Detection 目录下运行
python scripts/infer_video.py `
  --source ../classroom.mp4 `
  --config configs/pipeline.yaml `
  --device 0 `
  --interval-sec 1.0 `
  --ppt-crop "100,50,800,600" `
  --save `
  --output artifacts/results/latest
```

> **提示**：不指定 `--det-weights` 时，默认下载使用官方 `yolo26m.pt`。请记得将 `"100,50,800,600"` (x1, y1, x2, y2) 替换为你真实视频里的 PPT 坐标。

### 6.2 冒烟测试（快速验证链路）

```bash
cd Class_Detection
python scripts/smoke_test.py --mock
```

### 6.3 独立测试 OCR (PPT文字提取)

你可以独立测试文字提取能力，不加载 YOLO/骨骼/动作识别等大模型，加快调试速度。

```bash
cd Class_Detection

# 1. 默认测试 (使用目录自带的 test.mp4)
python scripts/test_ocr_standalone.py

# 2. 指定其他视频
python scripts/test_ocr_standalone.py --source "/path/to/your/video.mp4"

# 3. 指定视频并限制 PPT 的坐标范围 (x1,y1,x2,y2)，能大幅提升识别率并减少杂乱输出
python scripts/test_ocr_standalone.py --source "test.mp4" --bbox "100,50,800,600"
```
> 提示：测试过程中，每一帧被识别的裁剪画面都会被保存在 `Class_Detection/artifacts/` 目录下。你可以查看这些图片，验证 `--bbox` 坐标是否刚好能框住 PPT 内容。

---

## 7. 训练产物与结果目录

检测训练默认输出：

```text
Class_Detection/artifacts/runs/detect/<run_name>/
├── weights/best.pt
├── weights/last.pt
├── args.yaml
├── results.csv
├── results.png
├── confusion_matrix.png
└── BoxPR_curve.png 等
```

推理默认输出：

```text
Class_Detection/artifacts/results/
├── <source>_<timestamp>.mp4
└── <source>_<timestamp>.json
```

---

## 8. 常见问题

1. 报错 `No module named ultralytics`：执行 `pip install -U ultralytics`。
2. 显存不足：优先降低 `--batch`（16->8->4），再降低 `--imgsz`（960->768->640）。
3. CUDA `device-side assert triggered`：
   - 先检查标签类别是否越界；
   - 再检查坐标是否越界；
   - 触发后请重启训练进程，避免使用已污染的 CUDA 上下文。
4. 目录路径错误：建议始终先 `cd Class_Detection` 后执行脚本。

---

## 9. 类别映射表 (双模型架构)

系统当前采用双模型架构，YOLO 直接负责行为的初步分类：

### 9.1 学生行为模型 (Behavior Model, 7 类)
```text
全局 ID → 类别名称
  0: write       (写字)
  1: read        (阅读)
  2: lookup      (抬头听课)
  3: turn_head   (转头/注意力不集中)
  4: raise_hand  (举手)
  5: stand       (站立)
  6: discuss     (讨论)
```
- **学生角色 (0-6)**：所有 0-6 类的检测框均划分为“学生”，配合 Gaze 与动作规则计算最终专注度 CAS 分数。

### 9.2 环境模型 (Env Model, 沿用老模型 3 类)
```text
全局 ID → 类别名称
  0: student       (过滤，不使用)
  1: teacher       (教师)
  2: screen_board  (屏幕/黑板)
```
- **环境角色 (2)**：仅提取 ID=2 的 `screen_board`，自动送入 OCR 知识点提取模块。
