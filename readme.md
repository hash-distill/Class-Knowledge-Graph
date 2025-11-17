## 课堂智能视觉识别系统



### 项目概述

本系统负责完成课堂场景下的两大核心视觉识别任务：

1. **PPT内容识别**：实时提取教学PPT中的知识点，自动构建课堂知识图谱
2. **学生状态识别**：多维度评估学生课堂参与度，生成学习画像

系统由张同学负责开发与维护，为东北师范大学"中央高校基本科研业务专项资金"本科生项目提供核心视觉感知能力。

### 技术栈详解

#### 任务一：PPT文字识别



| 模块           | 技术方案                  | 作用                |
| :------------- | :------------------------ | :------------------ |
| **输入捕获**   | OpenCV VideoCapture / MSS | 摄像头/屏幕实时捕获 |
| **图像预处理** | OpenCV几何变换            | 透视矫正、边缘检测  |
| **OCR引擎**    | **PaddleOCR v4**          | 中文识别准确率95%+  |
| **文本处理**   | SimHash + 正则表达式      | 知识点提取 + 去重   |
| **输出**       | JSON Lines格式            | 结构化知识流        |

#### 任务二：学生状态识别



| 模块           | 技术方案            | 作用               |
| :------------- | :------------------ | :----------------- |
| **人脸检测**   | MediaPipe BlazeFace | 30ms级实时检测     |
| **多目标跟踪** | DeepSort            | 跨帧学生ID保持     |
| **多任务学习** | **SAGE-Net**        | 头部姿态+视线+表情 |
| **行为识别**   | SlowFast 3D-CNN     | 举手/记笔记/趴下等 |
| **融合分析**   | 注意力机制          | 综合参与度评分     |

#### 环境要求



```yaml
操作系统: Ubuntu 20.04+ / Windows 10+
Python版本: 3.9.18 (必须)
CUDA版本: 11.8 (推荐)
GPU显存: 6GB+
系统内存: 16GB+
摄像头: 1080p分辨率，30fps
```





### 项目结构



#### 项目目录

```
vision_system/
├── README.md                          # 项目说明文档
├── requirements.txt                   # Python依赖
├── config.yaml                        # 配置文件
├── run_system.py                      # 一键启动脚本
├── train_sage.py                      # SAGE-Net训练脚本

├── ppt_recognition/                   # PPT识别模块
│   ├── main.py                        # 主程序入口
│   └── processor.py                   # 图像处理与OCR
│
├── student_engagement/                 # SAGE-Net学生状态模块
│   ├── main.py                         # 主程序入口
│   ├── analyzer.py                     # SAGE分析引擎
│   └── models/
│       └── multitask_net.py            # SAGE-Net模型定义
│
├── utils/                             # 工具模块
│   └── tracker.py                     # 多目标跟踪封装
│
├── models/                            # 预训练模型
│   ├── download_models.sh             # 模型下载脚本
│   └── sage_net.pth                   # SAGE-Net权重文件（需下载或训练）
│
└── outputs/                           # 输出目录
    ├── knowledge_points.jsonl         # PPT识别结果
    └── engagement_log.jsonl           # 学生状态日志
```



#### 配置文件

`config.yaml`

```yaml

# PPT识别参数
ppt_detection:
  confidence_threshold: 0.7      # OCR置信度阈值
  simhash_threshold: 5           # 文本相似度阈值
  stable_frame_count: 30         # 内容稳定帧数

# 学生状态参数
engagement:
  model_path: "models/engagement_net.pth"
  gaze_threshold: 0.4            # 视线偏离阈值
  head_yaw_threshold: 30         # 头部偏转角度阈值(度)
  engagement_buffer_size: 50     # 参与度缓存帧数
  max_num_faces: 10              # 最大检测人数

# 摄像头配置
cameras:
  - id: 0
    position: "front"
    resolution: [1280, 720]
    fps: 30
  - id: 1
    position: "back"
    resolution: [1280, 720]
    fps: 30

# 输出配置
output:
  save_video: false              # 是否保存视频
  save_interval: 1.0             # 结果保存间隔(秒)
  log_level: "INFO"              # 日志级别
```



### 快速开始

#### 任务一：PPT内容识别


```bash
cd ppt_recognition

# 从摄像头识别
python main.py --source 0

# 从屏幕捕获识别
python main.py --source screen

# 从视频文件识别
python main.py --source path/to/lecture.mp4
```

**输出示例**：


```jsonl
{"timestamp": 1700191234.567, "knowledge_point": "definition: 函数是一一映射关系", "raw_text": "函数的定义：对于每一个x∈D，都有唯一y∈R与之对应", "confidence": 0.93}
{"timestamp": 1700191245.123, "knowledge_point": "theorem: 勾股定理 a²+b²=c²", "raw_text": "勾股定理：在直角三角形中，两条直角边的平方和等于斜边的平方", "confidence": 0.91}
```

------

#### 任务二：学生状态识别



```bash
cd student_engagement

# 单摄像头实时分析（带可视化）
python main.py --camera_id 0 --display

# 后台运行（无界面）
python main.py --camera_id 0

# 从配置文件启动多摄像头
python main.py --config config.yaml
```

**输出示例**：



```jsonl
{"timestamp": 1700191234.567, "students": [
    {"student_id": 1, "engagement_score": 82.5, "attention": "focused", "emotion": "专注"},
    {"student_id": 2, "engagement_score": 45.3, "attention": "distracted", "emotion": "疲惫"}
]}
```

------

### 性能指标



| 任务             | 模型      | 帧率   | 显存占用 | 延迟  | 准确率 |
| :--------------- | :-------- | :----- | :------- | :---- | :----- |
| PPT识别          | PaddleOCR | 15 FPS | 2GB      | 200ms | 95%+   |
| 状态识别(单目标) | SAGE-Net  | 30 FPS | 1GB      | 33ms  | 88%+   |
| 状态识别(10目标) | SAGE-Net  | 25 FPS | 4GB      | 40ms  | 85%+   |



### 模型训练



## 模型训练

如需微调学生状态识别模型，请按以下步骤操作：

### 1. 数据集准备

```
data/
  ├── train/
  │   ├── images/          # 裁剪的人脸图像
  │   └── labels.csv       # 标注: student_id, engagement_score, emotion, head_pose
  └── val/
      ├── images/
      └── labels.csv
```

### 2. 启动训练

```bash
cd student_engagement
python train.py \
  --data_dir data/ \
  --batch_size 32 \
  --epochs 50 \
  --lr 1e-4 \
  --device cuda
```

### 3. 模型评估

```bash
python evaluate.py --model_path outputs/best_model.pth --test_dir data/test/
```
