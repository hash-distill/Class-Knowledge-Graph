# 智启教学：内容与课堂行为联动

2026-10-26 实际运行版。主工作台保留语音、屏幕课件、知识图谱、独立课件导入和证据问答，新增同源行为分析。Python API 不加载视觉模型；视觉推理通过独立脚本离线生成，再经校验发布。

## 打开已经准备好的演示

在 `4C/课堂复盘` 执行：

```powershell
.\start_trained_demo.ps1
```

当前默认入口为同课三模态中文演示：<http://127.0.0.1:5175/?lesson=unified-chemistry-001-9&view=review>。同一段 90 秒中文录像提供转写、真实投影和可见行为；原中文数学、语文及化学示例仍可切换，英文课留在归档中。

行为演示：<http://127.0.0.1:5175/?lesson=ts-chemistry-001-9>。化学课堂全长90秒，约40秒处有读写行为及有效规则分数；课件镜头保留未知。数学课堂 <http://127.0.0.1:5175/?lesson=ts-math-006-32> 全长52秒，未取得足够时序证据，曲线保持空白，仍展示真实举手标签及原图。

启动脚本复用已运行的服务；启动本地模型、API、前端并核验演示数据。CPU大模型可用 `-LlmBackend cpu`；代码更新后用 `-RestartApi`。未下载模型或安装依赖的新机器需先按现有README准备环境，脚本不自动下载数据。原始视频仍从本地数据集读取，迁移机器时需更新 `configs/lessons.json`。

工作台已拆为“课堂总览、语音与课件、学生行为、知识图谱、复盘报告”五个独立视图。共用一个播放器，切换保留播放时间；助手按需展开。行为页另分概览、热图、原帧、模型与边界，避免全部铺成长页面。完整数据导出保留原始记录。

## 三种模型输出的含义

| 输出 | 当前输入和任务 | 与当前录像的关系 |
|---|---|---|
| 可见行为 | 现有预训练YOLO人体/姿态与阅读、书写、举手检测，外加时序规则 | 对当前原始录像实际推理；本次没有重新训练这套检测器 |
| TS教学情境 | 冻结ImageNet MobileNetV3 Small，每段16帧，特征均值和标准差，训练8类多标签线性头 | RGB可直接推理；只给整段粗粒度预测，不能当逐秒标签 |
| ARIC训练记录 | 全景CLIP特征 `(1,401,768)` 均值池化+L2归一化，训练多标签头 | 离线图像级“某类行为是否存在”；当前不接入逐人视频预测 |

检测对象没有确认师生身份。规则参与指数为0–100的未校准规则量，不是心理积极度或知识掌握；没有对象、证据不足、切镜头后的等待期保持未知。规则分数覆盖率的分子为**规则分数非空**的对象观测数，并非即时标签已识别数量。跨帧累计是人次，不是独立人数。轨迹只在同镜头内关联。

## 本次真实训练和评测

| 项目 | TS | ARIC |
|---|---|---|
| 本地可用范围 | 70个公开视频片段，8个原录像组，共约1.365小时 | 现有全景特征及框标注，原始RGB/个体特征缺失 |
| 使用规模 | 70段均提取特征；44训练、10验证、16测试 | 确定性抽样2048张；1228训练、410验证、410测试 |
| 划分单位 | 原录像组；数学006+美术001测试，语文001验证 | 原始图像ID；原提供索引跨划分重叠，故重新分组 |
| 分类头 | 8类多标签线性头 | 25个训练支持充足的标签，其余7类排除 |
| 测试Macro-F1 | 0.227814 | 0.450011 |
| 测试Micro-F1 | 当前报告未计算 | 0.686210 |
| 测试mAP | 0.460947 | 0.472439 |
| 对照 | 训练先验Macro-F1 0.119048 | 全阳性Macro-F1 0.284397；训练先验Macro-F1 0.122266 |
| 实测耗时 | 本地CPU提取1120帧约151.56秒；A6000分类头约1.97秒 | 本地准备约369.49秒，其中真实特征缓存约350.87秒；CPU头约1.55秒；服务器CPU重跑约0.64秒 |

分类头规模很小，秒级拟合不包括准备环境、读取压缩包、视频处理或检测器微调。TS实际使用服务器的单张RTX A6000，物理GPU3映射为进程内`cuda:0`；ARIC现有头为NumPy CPU计算，没有声称使用GPU加速。评测文件分别位于 `artifacts/learning/ts-context/metrics.json`、`artifacts/learning/aric/metrics.json`；远端ARIC复跑为 `remote_metrics.json`。

TS只是公开70段样例，未取得973段全集。小测试组及TS低Macro-F1说明它是可运行基线；不能宣传成熟准确率。化学演示来源属于TS**训练组**，不是独立测试；页面和问答证据注明此点。数学来源为测试组，整段预测的表现也不能替代逐人行为验证。ARIC的taking_bottle等弱类保留在报告中，没有隐藏失败类别。

所有训练归一化仅使用训练集，阈值仅用验证集；TS不输入参考转写、行为描述或目标标签文本。ARIC不输入描述文本。`model.npz`禁用pickle读取；TS读取校验训练元数据中的SHA-256。训练数据与个人课堂媒体不随开源源码分发，发布模型前仍需按相应数据和预训练权重的授权范围确认。

## 可复跑入口

以下命令在 `4C/课堂复盘` 执行。训练环境需要NumPy、PyTorch、TorchVision、OpenCV；当前本地视觉环境已具备。已有媒体环境负责ASR/OCR与生成课程包。

```powershell
# TS原始RGB特征，冻结编码器；使用新的output目录保留既有结果
..\vision\.venv-inference\Scripts\python.exe scripts/train_ts.py --dataset 'E:\Research\Class Detection Dataset\TS-Interact' --output artifacts/learning/ts-retrain --device cpu --extract-only
# 同一缓存及manifest可拷至A6000，再运行下行；CUDA_VISIBLE_DEVICES由服务器使用者选空闲卡
python scripts/train_ts.py --features artifacts/learning/ts-retrain/features.npz --output artifacts/learning/ts-retrain --device cuda:0 --epochs 300 --seed 42

# ARIC入口和合同详见docs/ARIC_TRAINING.md；已有portable bundle可不访问大归档复跑
python scripts/train_aric.py --from-bundle artifacts/learning/aric/training_bundle.npz --output artifacts/learning/aric-rerun

# 原视频全流程：先准备真实ASR/OCR，再构建课程，再分析行为并发布完整叠加包
.venv-media\Scripts\python.exe scripts/prepare_media.py --video '<原始录像.mp4>' --output artifacts/my-lesson --mode all --roi 0 0 1 1 --sample-interval 5
.venv-media\Scripts\python.exe scripts/build_review.py --lesson-id my-lesson --video '<原始录像.mp4>' --media-dir artifacts/my-lesson --title '课堂行为与内容联动'
..\vision\.venv-inference\Scripts\python.exe scripts/analyze_activity.py --video '<原始录像.mp4>' --lesson-id my-lesson --media-dir artifacts/my-lesson --interval .5 --imgsz 960 --device cpu --context-model artifacts/learning/ts-context --aric-report artifacts/learning/aric/metrics.json
```

`--enrich-only`可对既有完整`activity.json`追加/更新TS模型及ARIC研究记录，不重新运行人体检测；仍验证原视频哈希、帧图像和时间。课程重建、课件导入/移除保留有效同源活动叠加。原媒体与原始ASR保留。

## 验收

```powershell
# 课堂复盘目录
.venv-media\Scripts\python.exe -m pytest tests -q
..\vision\.venv-inference\Scripts\python.exe -m pytest tests/test_ts_learning.py -q
# frontend目录
npm run test:review
npm run test:activity
npm run build
python scripts/verify-proofread-students.py
```

浏览器脚本需要Playwright及Chromium，直接访问真实API、媒体和本地大模型，不设模拟路由。输出在前端 `output/playwright/trained-activity-real`，包括原图点击、未知状态、热图、导出、行为引用回跳、手机布局的验收记录和截图。

## 中文示例补充（2026-10-27）

语文示例为同一原视频的82秒完整片段，编号 `ts-chinese-001-2`。已运行165帧行为抽样、6个检测镜头段；累计3110人次检测、46人次具有规则分数。课件使用中文识别、原分辨率和完整画面，得到6个页段、9个可定位概念。界面过滤明显的文字识别乱码行；原始识别、来源与模型元数据保留在完整数据中。

该片段默认语音检测只保留约4.352秒，未得到可靠转写。关闭语音过滤的前25秒诊断仍未输出可靠文字，因此没有补写台词。以 `--allow-no-speech` 显式生成画面证据课堂，语音状态为“未识别”，问答依据实际课件和行为。主示例继续使用有完整中文语音的《正弦定理》。

启动可指定 `-LessonId ts-chinese-001-2` 或 `-LessonId ts-chemistry-001-9`。默认前端端口改为5175，因为5173目前属于另一个项目。`scripts/verify-chinese-demo.py` 核验默认中文课、真实语音问答、语文画面证据、中文引用和导出；所有原始证据不因中文化被改写。

## 同课三模态演示操作

1. 打开默认“课堂总览”，沿同一时间查看三种线索。约40秒可看实际阅读、书写及规则分数；43.5～47.5秒可看中文投影，此时人物不可见，行为保持未知。不能把不同镜头的状态拼成同时发生。
2. 切换“语音与课件”，查看28段课件词汇提示的机器转写。原36段无提示转写保存在完整数据中；两者均未人工逐字核准。部分化学术语仍会误识。
3. 该页“独立课件”已导入由原45秒投影整理的配套文件；真实匹配仅覆盖43.5～47.5秒。第二页为来源说明，没有课堂时间。配套PPTX和PDF在 `artifacts/unified-demo-courseware`，它们是观测整理件，不是原作者原始课件。
4. “学生行为”查看曲线、热图和原帧；“知识图谱”只显示原模型候选中有实际课件字面支持的5个词，原12候选另存。规则指数不是心理积极度。
5. 打开“复盘助手”，可分别引用本课语音、课件和可见行为；点击引用回到共用播放器。“复盘报告”可导出完整数据，保留所有原始记录与来源。

当前浏览器验收脚本为前端 `scripts/verify-unified-workbench.py`，它直接使用真实API、原视频、本地模型，并通过浏览器重新导入配套PDF。已验证三类来源同一视频哈希、播放中导航、前进后退、保留搜索/对话、重导入、引用回跳、完整导出和手机布局。
