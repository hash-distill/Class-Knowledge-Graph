# 课堂知识图谱视觉识别模块

## 项目简介

本模块是 **课堂知识图谱国创项目** 的核心感知层组件，聚焦教育场景下的视觉信息智能解析。通过整合 **PaddleOCR 高精度文字识别** 与 **YOLO11 实时目标检测** 技术，实现两大核心功能：



1. 从课堂 PPT（图片 / 视频帧）中结构化提取知识点文本（支持公式、复杂排版）；

2. 实时识别学生课堂状态（举手、专注、走神等）并关联对应知识点；

3. 输出时间戳对齐的结构化数据，为上层知识图谱构建提供标准化视觉输入。

模块具备高兼容性、易部署、可扩展特性，适配真实课堂复杂环境（光照变化、多学生遮挡、PPT 快速切换等），助力国创项目实现 “教学内容 - 学生反馈” 的闭环分析。

## 技术栈选型



| 功能模块     | 核心技术 / 工具                       | 技术优势                                   |
| ------------ | ------------------------------------- | ------------------------------------------ |
| PPT 文字识别 | PaddleOCR 2.7.0、OpenCV               | 多语言支持、公式识别兼容、排版鲁棒性强     |
| 学生状态检测 | YOLO11（Ultralytics 8.2.22）、PyTorch | 实时推理（FPS≥30）、小目标检测精准、易微调 |
| 视频处理     | FFmpeg、OpenCV                        | 高效帧提取、格式兼容（MP4/AVI/FLV 等）     |
| 数据结构化   | Python、JSON                          | 轻量易解析，适配知识图谱数据格式           |
| 环境依赖     | Python 3.11、CUDA 12.0                | 兼顾兼容性与 GPU 加速性能                  |

## 环境快速搭建

### 前置要求



* 操作系统：         Windows 11

* 硬件配置：

  

  CPU：AMD Ryzen 7 8845H w            

  GPU：4060

  


* **我的环境**



![](https://raw.githubusercontent.com/hash-distill/PicGo-Repo/master/202511151556352.png)



CUDA版本: 11.7  Pytorch版本：2.0.1    Cudnn版本：90100



### 步骤 1：创建虚拟环境（推荐）



```
\# Conda 方式（推荐，避免依赖冲突）

conda create -n classroom-vision python=3.10 -y

conda activate classroom-vision

\# 或 Venv 方式

python -m venv venv

\# Windows 激活：venv\Scripts\activate

\# Linux/macOS 激活：source venv/bin/activate
```

### 步骤 2：安装依赖库



```
\# 1. 安装 PaddlePaddle（GPU 版，无 GPU 替换为 CPU 版）

\# GPU 版（需 CUDA 11.8 环境）

pip install paddlepaddle-gpu==2.7.0.post118 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html

\# 2. 安装核心视觉库

pip install paddleocr==2.7.0 ultralytics==8.2.22

\# 3. 安装辅助工具库

pip install opencv-python numpy pandas pillow ffmpeg-python
```

## 项目结构说明



```
classroom-vision/

├── paddle\_ppt\_ocr.py                # PPT 文字识别核心脚本（支持命令行调用）

├── student\_status\_detector.py       # 学生状态检测核心脚本（YOLO11 驱动）

├── classroom\_vision\_integration.py  # 数据集成脚本（关联 PPT 与学生状态）

├── models/                          # 模型目录（存放自定义训练权重）

│   └── yolov11\_student\_best.pt      # 学生状态检测微调模型（示例）

├── data/                            # 测试数据目录（可替换为实际数据）

│   ├── test\_ppt.jpg                 # 示例 PPT 图片

│   └── classroom\_lecture.mp4        # 示例课堂视频（含 PPT 与学生画面）

├── outputs/                         # 输出目录（自动生成）

│   ├── ocr\_results.json             # PPT 文字识别结果

│   ├── student\_detections.json      # 学生状态检测结果

│   ├── integrated\_results.json      # 整合结果（知识图谱输入数据）

│   └── annotated\_video.mp4          # 学生状态标注视频（可视化验证）

└── README.md                        # 项目说明文档（本文件）
```

## 核心功能使用指南

### 功能 1：PPT 文字识别（paddle\_ppt\_ocr.py）

从单张 PPT 图片或课堂视频中提取结构化文字，支持公式、多段落排版，输出文字内容、位置坐标及时间戳。

#### 命令行参数详情



| 参数         | 类型  | 是否必选 | 说明                                      | 示例值                    |
| ---------- | --- | ---- | --------------------------------------- | ---------------------- |
| --type     | str | 是    | 处理类型：image（单张图片）/video（视频）              | image                  |
| --input    | str | 是    | 输入文件路径（绝对路径 / 相对路径）                     | data/test\_ppt.jpg     |
| --interval | int | 否    | 视频帧提取间隔（秒），默认 5 秒                       | 3（快速切换 PPT 推荐）         |
| --use-gpu  | 开关  | 否    | 启用 GPU 加速（默认 CPU 处理）                    | 无需赋值，加参数即启用            |
| --output   | str | 否    | 输出 JSON 路径，默认 outputs/ocr\_results.json | outputs/ppt\_text.json |

#### 使用示例

##### 示例 1：识别单张 PPT 图片（GPU 加速）



```
python paddle\_ppt\_ocr.py --type image --input "data/test\_ppt.jpg" --use-gpu --output "outputs/ppt\_single.json"
```

##### 示例 2：从课堂视频提取 PPT 文字（每 3 秒 1 帧）



```
python paddle\_ppt\_ocr.py --type video --input "data/classroom\_lecture.mp4" --interval 3 --use-gpu
```

### 功能 2：学生状态检测（student\_status\_detector.py）

基于 YOLO11 模型实时识别学生课堂状态，支持 4 类核心状态（可扩展），输出目标位置、类别、置信度及时间戳，可选生成标注视频用于可视化验证。

#### 支持的学生状态类别



| 类别名称    | 标签（class）   | 说明                |
| ------- | ----------- | ----------------- |
| 举手      | hand\_up    | 学生举手示意（积极互动）      |
| 专注听讲    | focus       | 抬头注视前方（PPT / 教师）  |
| 走神 / 低头 | distracted  | 低头看手机 / 课本（非专注状态） |
| 小组讨论    | group\_chat | 与周边同学互动交流（积极状态）   |

#### 命令行参数详情



| 参数            | 类型    | 是否必选 | 说明                                               | 示例值                              |
| ------------- | ----- | ---- | ------------------------------------------------ | -------------------------------- |
| --input       | str   | 是    | 输入课堂视频路径                                         | data/classroom\_lecture.mp4      |
| --model       | str   | 否    | YOLO11 模型路径，默认使用预训练微调模型                          | models/yolov11\_student\_best.pt |
| --conf        | float | 否    | 置信度阈值（过滤低置信结果），默认 0.5                            | 0.6（减少误检）                        |
| --use-gpu     | 开关    | 否    | 启用 GPU 加速                                        | 加参数即启用                           |
| --output-json | str   | 否    | 检测结果 JSON 路径，默认 outputs/student\_detections.json | outputs/student\_status.json     |
| --output-vid  | str   | 否    | 标注视频输出路径，默认 outputs/annotated\_video.mp4         | outputs/student\_vid.mp4         |

#### 使用示例



```
python student\_status\_detector.py --input "data/classroom\_lecture.mp4" --conf 0.6 --use-gpu --output-vid "outputs/student\_annotated.mp4"
```

### 功能 3：数据集成（classroom\_vision\_integration.py）

按时间戳对齐 PPT 文字识别结果与学生状态检测结果，生成结构化 JSON 数据，可直接作为知识图谱的 “知识点 - 学生反馈” 输入。

#### 命令行参数详情



| 参数             | 类型  | 是否必选 | 说明                                                 | 示例值                          |
| -------------- | --- | ---- | -------------------------------------------------- | ---------------------------- |
| --video        | str | 是    | 输入课堂视频路径（与前两步一致）                                   | data/classroom\_lecture.mp4  |
| --ocr-json     | str | 否    | PPT 识别结果 JSON 路径，默认 outputs/ocr\_results.json      | outputs/ppt\_text.json       |
| --student-json | str | 否    | 学生检测结果 JSON 路径，默认 outputs/student\_detections.json | outputs/student\_status.json |
| --output       | str | 否    | 整合结果输出路径，默认 outputs/integrated\_results.json       | outputs/kg\_input.json       |

#### 使用示例



```
python classroom\_vision\_integration.py --video "data/classroom\_lecture.mp4" --output "outputs/kg\_input.json"
```

## 输出结果格式说明

### 1. 整合结果（知识图谱输入数据）

`integrated_results.json` 是核心输出，格式如下（时间戳对齐知识点与学生状态）：



```
\[

&#x20; {

&#x20;   "timestamp": 60.2,  // 视频时间戳（秒）

&#x20;   "frame\_idx": 1806,  // 对应视频帧索引

&#x20;   "ppt\_knowledge": \[  // PPT 提取的知识点文本

&#x20;     {

&#x20;       "text": "勾股定理：a² + b² = c²",

&#x20;       "box": \[120, 80, 350, 120],  // 文字位置（x1,y1,x2,y2）

&#x20;       "confidence": 0.98

&#x20;     }

&#x20;   ],

&#x20;   "student\_feedback": {  // 学生状态统计

&#x20;     "total\_students": 32,  // 检测到的学生总数

&#x20;     "active\_status": {     // 积极状态统计（举手+专注+讨论）

&#x20;       "count": 25,

&#x20;       "rate": 0.781

&#x20;     },

&#x20;     "status\_details": \[    // 单学生状态详情（部分）

&#x20;       {

&#x20;         "box": \[200, 300, 250, 450],

&#x20;         "class": "focus",

&#x20;         "confidence": 0.92

&#x20;       }

&#x20;     ]

&#x20;   }

&#x20; }

]
```

### 2. 标注视频说明

`annotated_video.mp4` 中会用不同颜色框标注学生状态：



* 绿色框：hand\_up（举手）

* 蓝色框：focus（专注）

* 黄色框：group\_chat（讨论）

* 红色框：distracted（走神）

可直接打开视频验证识别准确性，便于调试优化。

## 模型优化与扩展指南

### 1. YOLO11 学生状态模型微调

若默认模型在特定课堂场景（如低光照、特殊座位排列）识别效果不佳，可通过以下步骤微调：



1. 用 Label Studio 标注自定义数据集（格式：COCO）；

2. 编写训练配置文件 `student_status.yaml`（参考 Ultralytics 文档）；

3. 执行训练命令：



```
yolo train model=yolov11n.pt data=student\_status.yaml epochs=50 imgsz=640 batch=16 device=0
```



1. 将训练生成的 `best.pt` 放入 `models/` 目录，运行时通过 `--model` 参数指定。

### 2. PPT 文字识别优化



* 公式识别增强：替换 PaddleOCR 模型为公式专用模型，修改 `paddle_ppt_ocr.py` 中初始化代码：



```
self.ocr = PaddleOCR(lang="ch", use\_gpu=True, show\_log=False, det\_model\_dir="path/to/formula\_det", rec\_model\_dir="path/to/formula\_rec")
```



* 低清晰度 PPT 优化：在 `process_single_image` 方法中添加图像增强逻辑（如对比度提升、降噪）。

## 国创项目集成建议



1. **数据对接**：将 `integrated_results.json` 作为输入，通过 Neo4j Python 驱动直接写入知识图谱，关联 “知识点”“学生”“课程” 实体；

2. **实时部署**：结合 Flask/FastAPI 封装为 HTTP 接口，对接课堂直播流（如 RTMP 流），实现实时分析；

3. **性能优化**：对视频进行分帧并行处理，或降低分辨率（如 720p），平衡速度与精度；

4. **隐私保护**：处理学生视频时，可通过 OpenCV 模糊面部区域，或仅输出状态统计数据（不保留个体特征）。

## 常见问题排查



| 问题现象                 | 可能原因               | 解决方案                                              |
| -------------------- | ------------------ | ------------------------------------------------- |
| PaddleOCR 导入报错       | PaddlePaddle 版本不兼容 | 卸载后重新安装指定版本：`pip install paddlepaddle-gpu==2.6.0` |
| YOLO11 推理速度慢（<5 FPS） | 未启用 GPU 加速或显存不足    | 检查 CUDA 配置，或使用轻量模型 `yolov11n.pt`                  |
| PPT 文字识别漏检公式         | 未使用公式专用模型          | 参考 “模型优化” 部分配置公式模型                                |
| 学生状态误检率高             | 置信度阈值过低或场景不匹配      | 提高 `--conf` 参数（如 0.6），或微调模型                       |

## 项目维护信息



* 项目负责人：\[你的姓名 / 团队名称]

* 技术支持：\[你的邮箱]

* 版本迭代：v1.0（基础功能版，适配国创项目一期需求）

* 更新日志：2024-XX-XX 初始版本发布，支持 PPT 识别与学生状态检测核心功能

## 开源声明

本模块基于以下开源项目二次开发，遵循对应开源协议：



* PaddleOCR：Apache License 2.0（[https://github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)）

* YOLO11（Ultralytics）：GPL-3.0 License（[https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)）

