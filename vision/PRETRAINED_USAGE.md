# 免训练课堂行为观察：直接运行

2026-10-25：默认入口已改为公开权重推理，未执行模型训练。已下载两份固定权重，已创建独立 `.venv-inference`，避免全局 Anaconda 的 OpenMP 冲突。

## 现在接入了什么

| 环节 | 实现 | 输出边界 |
|---|---|---|
| 人物、姿态与轨迹 | 官方 YOLO26n-Pose + ByteTrack | 17 点姿态与当前录像中的轨迹 ID；不识别姓名，不自动区分师生 |
| 课堂行为 | SCB 场景公开 YOLOv9e 权重 | 仅举手、阅读、书写三类，实际类别从 checkpoint 核验 |
| 行为归属 | 行为框与人物框一对一匹配 | 无法匹配时保留未知；姿态举手规则单独标记来源 |
| 时间融合 | 4 秒证据窗口，按真实秒数积分 | 质量、有效持续时间与教学任务共同决定是否给出规则估计 |
| 教学内容 | 用户提供带秒数的任务/知识点标注；OCR 可选 | 不自动生成知识点，不用行为分数推断知识掌握 |
| 展示 | 真实 JSON 导入、帧回放、时间曲线、对象热图、来源查询 | 模拟模式与真实模式分开；未知保持 null/— |

**规则指数不是训练出的学生投入概率。** 三类行为主要提供积极参与线索；当前并没有可靠的“低投入”分类器。未知对象不参与均值，因此高指数可能仅代表少数有证据的对象，必须同时查看覆盖率。没有用人工标注的目标课堂验证准确率，不能宣称已达到某个准确率或已经提升。

## 本机快速运行

PowerShell 在本目录运行；相对路径按当前工作目录解释：

```powershell
cd 'D:\全球校园人工智能精英大赛\4C\视觉模型'
./run_pretrained.ps1 -Source '../frontend/public/data/visual/latest/test.mp4' -MaxFrames 35 -Device cpu
```

替换为自己的录像，并去掉 `-MaxFrames 35`，即可处理完整文件。默认约每 0.2 秒分析一帧、约每秒保存一个快照，末帧也保存；分析时间会按视频 FPS 取最接近的帧。处理间隔和导出间隔互相独立。

```powershell
./run_pretrained.ps1 -Source 'D:\录像\课堂.mp4' -Task lecture -StudentRoi '0.05,0.1,0.95,1' -AnnotatedVideo
```

- `-Task`：`unspecified / lecture / exercise / discussion / presentation`，为规则提供教学情境，不改变已训练模型。默认 unspecified。
- `-StudentRoi`：归一化 `x1,y1,x2,y2`，只统计中心落在区域内的检测对象；需按实际机位设置。默认整幅画面，可能包含老师和其他人物。
- `-AnnotatedVideo`：额外输出带框视频；默认只输出 JSON 和抽样帧图。
- `-PoseOnly`：只跑人物姿态和几何举手提示，用于设备排查；会失去阅读/书写识别，不能称为完整方案。
- `-SnapshotIntervalSec 0`：保存每个分析帧，适合短视频验证。默认 1 秒减小文件量。
- `-Device auto`：有可用 CUDA 时选择 `cuda:0`，否则 CPU。当前安装的是 CPU PyTorch，已验证 CPU 路径；CUDA 执行尚未实测。明确指定 GPU 但环境不可用时会报错，不静默换设备。

新结果写入 `Class_Detection/artifacts/results/pretrained_时间戳/`，不会覆盖旧录像和旧结果：

```text
snapshots.json             # 前端可导入的 JSON 数组
snapshots_002.json         # 长录像自动分段，后续编号递增
frames/frame_0000001.jpg   # 抽样原图
observations.mp4           # 仅请求 AnnotatedVideo 时生成
run_manifest.json          # 成功完成标记、全部分段、模型摘要、配置和统计
```

单个 JSON 段上限 16 MiB，兼容前端 25 MB、2 万帧、30 万对象记录的导入限制。前端一次载入一段；秒数保持原录像时间。运行清单中的 `analyzed_frames` 是分析帧数，`snapshots` 是导出帧数，人数统计也是分析帧上的重复观测次数，不是独立学生人数。异常时保留 `.partial` 排查，**只有 `run_manifest.json` 中 `completed=true` 才表示整次任务完成**。

## 前端查看

在 `4C/frontend` 运行 `npm run dev`，点击“导入推理 JSON”选择某个 `snapshots*.json`。文件仅本机读取；浏览器不能通过本地 JSON 自动读取旁边的图片。

需要帧图时，把本次结果目录整体复制到 `Front/public/data/visual/` 下一个新名称，再用同源地址加载：

```text
http://127.0.0.1:5173/?data=/data/visual/你的结果目录/snapshots.json
```

本机已提供一次 35 帧联调结果：

```text
http://127.0.0.1:5173/?data=/data/visual/pretrained-v2/snapshots.json
```

该记录来自项目已有演示录像的真实模型推理，来源于录像并不等于独立测试集。页面默认进入录像首帧，连续证据尚未积累时分数为空；播放或拖动进度可查看后续估计。无知识标注时图谱保持空态。可展开“模型来源”查看完整 SHA-256、固定版本和来源链接。

## 教学任务与知识点对齐

手动核对录像时间后，建立 UTF-8 JSON，例如下列**仅演示格式**的时间段；不要未经核对直接当作真实课堂标注：

```json
[
  {"start_sec": 0, "end_sec": 120, "task": "lecture", "knowledge_point": "实际讲授的知识点"},
  {"start_sec": 120, "end_sec": 300, "task": "exercise"}
]
```

使用 `-ContextJson './context.json'`。区间为左闭右开，必须不重叠且秒数非负；未覆盖区间回到默认任务。知识点字段仅作内容标注，分数保持 null。当前默认未启用 ASR、PPT OCR、脸部视线或大语言模型，以避免未验证的依赖和推断进入结果。OCR 是可选旧模块，需要额外 PaddleOCR 环境，未纳入本次冒烟验收。

## 在另一台机器重建环境

当前已装环境无需重复执行。以下示例复建本次 CPU 路径；Python 3.11 和 uv 需可用：

```powershell
uv venv .venv-inference --python 3.11
uv pip install --python .venv-inference/Scripts/python.exe torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-inference/Scripts/python.exe -r requirements-inference.txt
& .venv-inference/Scripts/python.exe Class_Detection/scripts/fetch_pretrained.py
& .venv-inference/Scripts/python.exe Class_Detection/scripts/fetch_pretrained.py --verify-only
& .venv-inference/Scripts/python.exe -m pytest Class_Detection/tests -q
```

`requirements-inference.lock.txt` 记录本次已安装版本；不是 CUDA 环境锁。下载器固定字节大小、revision 和 SHA-256，推理前再次检查任务、类别与架构。遇到损坏文件会拒绝加载，显式 `--repair` 才尝试修复。

## 保留项与开源边界

原有代码、训练手册和公式保留为 legacy 对照，新推理默认不调用它们。旧配置是 `Class_Detection/configs/pipeline_legacy.yaml`；不要用旧公式分数与新规则指数直接混算。源代码备份在 `../模型优化方案/改造备份/`。

权重的固定地址、哈希、实际类别与许可记录见 [PRETRAINED_MODELS.md](Class_Detection/docs/PRETRAINED_MODELS.md)。行为权重可下载，但 SCB 上游有非商业限制；Space 的 MIT 元数据不覆盖所有上游权益。Ultralytics 也有其自身许可要求。因此本次接入不代表已取得可任意商业分发的授权。发布仓库时建议提交代码、清单和下载器，分别声明第三方来源；不要把演示录像和下载权重不加核对地一起打包。
