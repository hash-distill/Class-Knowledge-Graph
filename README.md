# 智启教学：融合知识图谱与多模态识别的课堂教学评估辅助系统

> 面向中小学真实课堂的智能教学评估系统，通过多模态 CV 感知与知识图谱联动，量化课堂教学效果。

---

## 0. 运行说明

本 README 仅保留系统设计与项目说明，训练/预测命令已统一迁移到独立手册：

- [TRAINING_AND_INFERENCE.md](TRAINING_AND_INFERENCE.md)

建议先阅读运行手册中的“SCB-Dataset 7 类标准流程”，再回到本 README 了解系统设计细节。

## 1. 项目概述

本项目面向真实线下中小学课堂，构建一套 **端到端的 CV 感知-评估 Pipeline**，解决三类核心问题：

运行与实验复现入口：
- 训练与预测手册：[TRAINING_AND_INFERENCE.md](TRAINING_AND_INFERENCE.md)
- 实施方案说明：[Class_Detection/docs/implementation_plan.md](Class_Detection/docs/implementation_plan.md)

1. **后排学生遮挡严重**：分辨率低、互相遮挡导致检测/追踪不稳定
2. **小样本动作识别难收敛**：课堂行为种类多但标注数据有限
3. **教学内容与学生反应的时间延迟**：知识点出现与学生反馈间存在 Time Lag，硬同步产生语义漂移

### 1.1 核心创新点

- **VSAM（Visual-Semantic Alignment Model）**：基于高斯先验的时间衰减窗口实现视觉与语义的零样本软对齐
- **双流感知架构**：YOLO26 双模型检测 (行为模型 + 环境模型) + YOLO26-Pose 姿态提取统一化
- **CAS/CTES 非线性评分体系**：max 融合避免指标稀释 + 方差指数惩罚检出极化课堂
- **PnP + 躯干 Fallback 视线估计**：替代不可靠的表情识别，兼容低分辨率场景

---

## 2. 技术路线总览

### 2.1 系统架构

```
视频输入 (1080p@25fps)
    │
    ▼
┌──────────────────────────────────┐
│  YOLO26 Detection + ByteTrack    │  ← yolo26m.pt + 内置追踪器
│  稳健 3 类检测 + Track ID       │
└──────────┬───────────────────────┘
           │
    ┌──────┴──────┐
    ▼              ▼
┌─────────┐  ┌──────────────┐
│YOLO26   │  │ Screen/Board │  ← 裁切 screen/blackboard 区域
│Pose     │  │ OCR (Paddle) │  ← 检测知识点文字变化
│(17 kpts)│  └──────┬───────┘
└────┬────┘         │
     │              │
     ├───────┐      │
     ▼       ▼      ▼
┌────────┐ ┌────┐ ┌──────────────┐
│ST-GCN  │ │PnP │ │ VSAM         │
│动作分类 │ │视线 │ │ 高斯时间对齐  │
│S_action│ │解算 │ │ Score_Ki      │
└───┬────┘ │S_g │ └──────┬───────┘
    │      └─┬──┘        │
    ▼        ▼            ▼
┌────────────────────────────────┐
│ CAS / CTES 评分引擎            │
│ → 结构化 JSON → 知识图谱更新   │
└────────────────────────────────┘
```

### 2.2 模型选型

| 功能 | 模型 | 权重文件 | 输出 |
|------|------|---------|------|
| 目标检测 | **YOLO26-M** (双模型) | `behavior_model`, `env_model` | 7 类学生行为 + 屏幕/黑板锚点 |
| 多目标追踪 | **ByteTrack** | ultralytics 内置 | 跨帧 Track ID |
| 姿态估计 | **YOLO26-N-Pose** | `yolo26n-pose.pt` | 17 COCO 关键点 (x, y, conf) |
| 动作分类 | **ST-GCN** | 自训练 | 行为标签 + 置信度 |
| 视线解算 | **PnP + Fallback** | cv2.solvePnP | 头部欧拉角 → 专注度分数 |
| 屏幕 OCR | **PaddleOCR** | 预训练 | 知识点文字锚点 |

### 2.3 YOLO26 特性说明

YOLO26 是 Ultralytics 于 2026 年 1 月发布的最新 YOLO 系列模型，具备以下核心特性：

- **End-to-End NMS-Free**：去除 NMS 后处理，降低推理延迟，硬件适配性更强
- **ProgLoss + STAL**：渐进损失平衡 + 小目标感知标签分配，提升拥挤场景精度
- **MuSGD 优化器**：融合 SGD 与 Muon 技术，收敛更快更稳定
- **多任务统一**：检测/分割/姿态/分类/OBB 共用同一框架

---

## 3. 数据集

### 3.1 主训练数据集：SCB-Dataset（双模型策略）

本项目底层检测采用 **双模型策略 (Dual-Model Architecture)** 来解决单数据集标签缺失的问题。

#### 1. 行为模型 (Behavior Model)
我们使用 **SCB-Dataset (原名 SCBehavior)** 来训练主干模型。该数据集精确标注了学生的 7 种日常课堂行为，使得模型能直接输出行为状态，免去复杂的后处理猜测。

```yaml
# configs/scb_dataset_yolo.yaml
names:
  0: write       # 写字
  1: read        # 阅读
  2: lookup      # 抬头听课
  3: turn_head   # 转头/分心
  4: raise_hand  # 举手
  5: stand       # 站立
  6: discuss     # 讨论
```

#### 2. 环境兜底模型 (Environment / Fallback Model)
由于行为数据集中不仅缺失黑板/屏幕的标注，**更严重的是漏标了绝大多数“安静听课”的学生**。如果只用行为模型，会导致安静的学生完全不被检测（被当作背景）。

因此，我们在底层架构中引入了 **原生 COCO 预训练模型 (如 `yolo26m.pt`)** 作为 `env_model`。
- **人员兜底**：利用原生 COCO 强大的找人能力，无死角检出所有 `person`。系统通过 NMS 融合，将未被行为模型覆盖的 person 自动分配为 `attending`（听课）状态，彻底解决数据集漏标缺陷。
- **环境提取**：自动将 COCO 的 `tv` 类别映射为 `screen_board`，专门用于提取 PPT 知识点截图 (OCR 锚点)。

#### 角色分组配置

| 角色 | 来源模型 | 类别 ID | 课堂含义 |
|------|----------|---------|----------|
| **学生 (特定行为)** | Behavior | 0~6 | YOLO 直接输出 7 类精确动作，配合姿态和 Gaze 综合打分 |
| **学生 (安静听课)** | Env (COCO) | 100 | NMS 融合后拯救的漏标学生，默认赋予 `attending` 状态 |
| **环境 (屏幕/黑板)** | Env (COCO) | 102 | 由 COCO 的 `tv` 类映射而来，提取 PPT 知识点截图 (OCR 锚点) |

### 3.2 辅助数据参考

| 数据集 | 用途 | 说明 |
|--------|------|------|
| **COCO-Keypoints** | YOLO26-Pose 预训练基础 | 已由 ultralytics 预训练权重覆盖 |
| **ARIC** (UESTC) | 行为分类体系参考 | 32 类活动分类标签体系；仅提供特征向量 |

---

## 4. CV 模块详细设计

### 4.1 检测与追踪模块 (`src/detector.py`)

**职责**：从视频帧中检测所有目标并维护跨帧追踪 ID。

```python
from src.detector import Detector

# 初始化双模型检测器
det_model = Detector(
    behavior_weights="artifacts/runs/detect/scbehavior_yolo26m/weights/best.pt",  # 7类行为
    env_weights="yolo26m.pt"  # 环境兜底模型 (COCO)
)

# 单帧双模型融合检测
results = det_model.detect_frame(frame)

# 带追踪的视频处理（内置 ByteTrack，仅对行为模型生效）
results = det_model.track_video(source=video_path, tracker="bytetrack.yaml")
```

**关键设计**：
- `persist=True`：跨帧保持 Track ID 一致性
- 目标分类后按角色分发：学生 BBox → 姿态估计流；屏幕/黑板 BBox → OCR 流
- 短时丢失轨迹续接：ByteTrack 内置 `track_buffer` 参数控制

### 4.2 姿态估计模块 (`src/pose.py`)

**职责**：对每个学生 BBox 提取 17 个 COCO 关键点。

```python
pose_model = YOLO("yolo26n-pose.pt")
results = pose_model(frame)
# keypoints shape: (N_persons, 17, 3)  → [x, y, confidence]
```

**17 个 COCO 关键点含义**：

```
0: 鼻子        5: 左肩       10: 右手腕     15: 左踝
1: 左眼        6: 右肩       11: 左髋       16: 右踝
2: 右眼        7: 左肘       12: 右髋
3: 左耳        8: 右肘       13: 左膝
4: 右耳        9: 左手腕     14: 右膝
```

**下游用途**：
- 关键点 0-4（面部）→ PnP 视线解算
- 关键点 5-6（肩部）+ 0（鼻）→ 躯干向量 Fallback
- 完整 17 点时序（T=30 帧）→ ST-GCN 动作分类

### 4.3 动作分类模块 (`src/action.py`)

**职责**：基于关键点时序判断学生当前行为类别。

**方案一：ST-GCN（主路径）**

ST-GCN（时空图卷积网络）将人体骨骼建模为图结构：
- **节点**：17 个关键点
- **空间边**：骨骼连接（如 肩→肘→腕）
- **时间边**：同一关键点在相邻帧间的连接

```
输入: (N, C, T, V, M)
N = batch_size
C = 3 (x, y, confidence)
T = 30 (时间窗口帧数)
V = 17 (关键点数)
M = 1 (单人)
```

**专注度评分三层优先级架构**（稳健 3 类系统）：

| 优先级 | 来源 | 输出标签 | S_action |
|---------|------|----------|:--------:|
| 1️⃣ ST-GCN | 关键点时序图卷积 | hand_raising / writing / attending 等 | 查 STGCN_ENGAGEMENT 表 |
| 2️⃣ 姿态规则 | 17 个 COCO 关键点 | hand_raising=0.95, writing=0.80, attending=0.75, looking_around=0.25, leaning=0.15 | 直接输出 |
| 3️⃣ 检测 Fallback | YOLO 输出 'student' | student | 0.70 (中性基线) |

> ℹ️ YOLO 仅负责框出人和屏幕，不再判断行为状态。行为分类由上述 3 层 fallback 链完成。

**姿态规则详解**（ST-GCN 未训练时启用，已在 `src/action.py` 实现）：

```python
def _rule_infer(window):
    """基于关键点运动统计特征的规则分类"""
    # 手腕高于肩膀 → 举手 (0.95)
    # 头部大幅低于肩膀 → 贴桌/睡觉 (0.15)
    # 头部运动幅度大 → 左右张望 (0.25)
    # 头部微低+平稳 → 读写 (0.80)
    # 头部正向+平稳 → 听课 (0.75)
```

### 4.4 视线估计模块 (`src/gaze.py`)

**职责**：计算学生视线方向和专注度分数 $S_{gaze}$。

#### PnP 主路径

利用面部关键点（鼻、眼、耳）映射至标准 3D 人脸模型，通过 `cv2.solvePnP` 解算欧拉角：

```python
# 3D 人脸模型（标准化坐标）
FACE_3D = np.array([
    [0.0,   0.0,    0.0],      # 鼻尖
    [-30.0,  40.0, -30.0],     # 左眼
    [ 30.0,  40.0, -30.0],     # 右眼
    [-60.0, -10.0, -50.0],     # 左耳
    [ 60.0, -10.0, -50.0],     # 右耳
])

# 从 YOLO26-Pose 关键点 [0,1,2,3,4] 取面部 2D 坐标
face_2d = keypoints[[0,1,2,3,4], :2].astype(np.float64)

# PnP 解算
success, rvec, tvec = cv2.solvePnP(FACE_3D, face_2d, camera_matrix, dist_coeffs)
# rvec → rotation matrix → Euler angles (pitch, yaw, roll)
```

**视线方向分类**：

| 视线方向 | Pitch 范围 | Yaw 范围 | $S_{gaze}$ |
|---------|-----------|---------|:---------:|
| 聚焦黑板/屏幕 | -15° ~ 15° | -20° ~ 20° | 0.85 ~ 1.0 |
| 低头专注（课桌） | -50° ~ -15° | -15° ~ 15° | 0.60 ~ 0.80 |
| 四处张望 | 其他 | \|yaw\| > 35° | 0.10 ~ 0.30 |

#### 躯干向量 Fallback

当面部关键点置信度 < 阈值时，使用肩部-鼻部向量估算朝向：

```python
def torso_fallback(kpts):
    """用肩-颈方向估算视线大致朝向"""
    mid_shoulder = (kpts[5] + kpts[6]) / 2  # 双肩中点
    nose = kpts[0]                          # 鼻尖
    direction = nose[:2] - mid_shoulder[:2]
    # 归一化后映射为专注度
    ...
```

### 4.5 VSAM 时间对齐模块 (`src/vsam.py`)

**职责**：将知识点出现时刻与学生反应进行软对齐。

**核心问题**：教师展示知识点 $K_i$ 后，学生群体反应存在 $\mu$ 秒的延迟（教育心理学规律）。硬同步（直接取同时刻状态）会导致语义漂移。

**解决方案**：高斯先验时间衰减窗口

```python
def gaussian_weight(t, t_ocr, mu=3.0, sigma=1.5):
    """
    t     : 当前时刻
    t_ocr : OCR 检测到知识点出现的时刻
    mu    : 先验延迟（学生反应峰值滞后秒数）
    sigma : 窗口宽度
    """
    center = t_ocr + mu
    return math.exp(-((t - center) ** 2) / (2 * sigma ** 2))

def score_knowledge_point(cas_values, timestamps, t_ocr, mu=3.0, sigma=1.5):
    """知识点吸收度 = 高斯加权 CAS 平均"""
    weights = [gaussian_weight(t, t_ocr, mu, sigma) for t in timestamps]
    return sum(w * s for w, s in zip(weights, cas_values)) / sum(weights)
```

**OCR 锚点触发逻辑**：
1. 每 N 帧对 screen/blackboard BBox 区域执行 PaddleOCR
2. 计算文本哈希差异（Levenshtein distance / hash 变化）
3. 超过阈值 → 判定为"知识点切换"→ 记录 $t_{ocr}$

### 4.6 评分引擎 (`src/scoring.py`)

#### 个体积极度 CAS

$$CAS = \max(w_1 \cdot S_{action},\ w_2 \cdot S_{gaze}) \cdot P_{negative}$$

**设计理由**：采用 `max` 而非加权平均：
- 学生积极记笔记（$S_{action}$ 高）但低头不看黑板（$S_{gaze}$ 低）→ 仍视为专注
- 学生静坐盯黑板（$S_{gaze}$ 高）但无明显动作（$S_{action}$ 低）→ 仍视为专注
- 加权平均会导致两个合理信号互相稀释
- $P_{negative}$：负面行为惩罚因子（使用手机/打哈欠等行为 × 0.5）

#### 课堂教学效果指数 CTES

$$CTES = \mu_{CAS} \cdot \exp(-\lambda \cdot \sigma_{CAS})$$

- $\mu_{CAS}$：全班 CAS 均值
- $\sigma_{CAS}$：全班 CAS 标准差
- $\lambda$：惩罚系数（默认 1.0）

**设计理由**：不仅看平均分，更将全班方差作为指数级惩罚。有效区分：
- 均值 0.7 + 方差 0.05 → CTES ≈ 0.55（理想状态：大多数学生专注）
- 均值 0.7 + 方差 0.30 → CTES ≈ 0.28（危险状态：两极分化严重）

---

## 5. 训练与预测（说明已迁移）

为避免 README 过长，训练、评估、推理的全部命令已统一整理至：

- [TRAINING_AND_INFERENCE.md](TRAINING_AND_INFERENCE.md)

该手册包含：
1. 环境与依赖安装。
2. SCB-5 稳健 3 类数据集构建（`build_scb5_unified.py`）。
3. 检测/姿态/ST-GCN 训练命令。
4. 评估、视频推理与冒烟测试命令。
5. 常见报错与排查建议。

---

## 6. 结构化输出协议

CV 模块通过 JSON Schema 向下游（知识图谱/大模型）传递评估数据：

```json
{
  "timestamp": "2026-05-10T10:15:30Z",
  "frame_id": 750,
  "frame_image_path": null,
  "knowledge_anchor": {
    "entity": "分数加减法",
    "trigger_time": "30.00s",
    "gaussian_weight": 0.95,
    "score_k": 0.82,
    "visual_score": 0.82
  },
  "classroom_metrics": {
    "ctes_score": 0.81,
    "mean_cas": 0.78,
    "std_cas": 0.15,
    "active_tracks": 35,
    "behavior_distribution": {
      "counts": {
        "read": 20,
        "write": 8,
        "discuss": 3,
        "hand_raising": 2,
        "talk": 1,
        "stand": 1
      }
    }
  },
  "student_states": [
    {
      "track_id": 7,
      "bbox": [100.0, 150.0, 200.0, 350.0],
      "action": {
        "label": "write",
        "confidence": 0.89,
        "engagement_score": 0.78,
        "det_confidence": 0.85,
        "source": "stgcn"
      },
      "gaze": {
        "pitch": -22.5,
        "yaw": 5.0,
        "roll": 0.0,
        "focus_score": 0.90,
        "focus_zone": "board_focus",
        "source": "pnp"
      },
      "cas": 0.89
    }
  ],
  "env_bboxes": [
    [50.0, 50.0, 800.0, 600.0]
  ]
}
```

---

## 7. 项目结构

```text
Class-Knowledge-Graph/
├── README.md
├── TRAINING_AND_INFERENCE.md   # 训练/评估/推理命令统一手册
├── requirements.txt
├── test_GPU.py
├── yolo26m.pt
├── SCB-Dataset/                # 用于行为检测的 SCB-Dataset (7类动作)
└── Class_Detection/
  ├── configs/
  ├── docs/
  │   └── implementation_plan.md
  ├── models/
  ├── scripts/
  ├── src/
  ├── tests/
  └── tools/
```

---

## 8. 评测指标体系

| 层级 | 指标 | 目标值 | 说明 |
|------|------|:------:|------|
| 感知层 | 检测 mAP@50 | ≥ 0.75 | YOLO26 在 SCB-Dataset 上的检测精度 |
| 感知层 | 追踪 IDF1 | ≥ 0.65 | ByteTrack 跨帧 ID 一致性 |
| 感知层 | 动作分类 F1 | ≥ 0.70 | ST-GCN 行为分类加权 F1 |
| 感知层 | 视线分类准确率 | ≥ 0.75 | PnP 视线方向三分类 |
| 对齐层 | Score_Ki Spearman 相关性 | ≥ 0.60 | VSAM 输出与人工观察趋势 |
| 系统层 | 端到端延迟 (1080p) | < 100ms | 单帧完整 Pipeline 处理时间 |
| 系统层 | 吞吐 FPS | ≥ 15 | 满足实时监控需求 |

---

## 9. 风险与预案

| 风险 | 预案 |
|------|------|
| 后排关键点缺失 | 启用躯干向量 Fallback，低置信帧降权 |
| OCR 错触发 | 翻页检测 + 文本变化阈值双门控 |
| 追踪 ID 抖动 | ByteTrack `track_buffer` + 短时续接策略 |
| ST-GCN 训练数据不足 | 降级为规则分类方法 |
| 现场演示异常 | 离线视频回放模式 + 容错脚本 |
| GPU 显存不足 | 降低 batch size / 切换 Nano 模型 |

---

## 10. 快速开始

按以下顺序执行即可：

1. 阅读并执行运行手册：[TRAINING_AND_INFERENCE.md](TRAINING_AND_INFERENCE.md)
2. 优先完成“SCB-Dataset 7类”首轮训练与验证。
3. 训练后使用 best.pt 进行评估与视频推理。

---

## 11. 环境依赖

完整依赖见 `requirements.txt`。核心依赖：

| 包 | 版本 | 用途 |
|----|------|------|
| ultralytics | ≥ 8.3.50 | YOLO26 检测/姿态/追踪 |
| torch | ≥ 2.5.0 (CUDA) | 深度学习后端 |
| opencv-python | ≥ 4.10.0 | PnP 解算 & 图像处理 |
| paddleocr | ≥ 2.8.1 | 屏幕文字识别 |
| scipy | ≥ 1.12.0 | 科学计算 |
| pydantic | ≥ 2.8.0 | 数据模型校验 |

---

## 12. 参考文献

- YOLO26: Ultralytics (2026). *YOLO26: Edge-First, NMS-Free Object Detection*. [docs.ultralytics.com](https://docs.ultralytics.com)
- ST-GCN: Yan et al. (2018). *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition*. AAAI 2018.
- ByteTrack: Zhang et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. ECCV 2022.
- RTMPose: Jiang et al. (2023). *RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose*. arXiv:2303.07399.
- SCB-Dataset: Whiffe et al. *Student Classroom Behavior Dataset*. [github.com/Whiffe/SCB-dataset](https://github.com/Whiffe/SCB-dataset)
- ARIC: Xu et al. (2024). *ARIC: An Activity Recognition Dataset in Classroom Surveillance Images*. arXiv:2410.12337.

---

## 13. 许可证

本项目用于学术研究和计算机设计大赛参赛使用。
