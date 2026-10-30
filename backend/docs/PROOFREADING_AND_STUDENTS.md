# 中文校对与当前学生状态

运行 `start_trained_demo.ps1` 后，打开 <http://127.0.0.1:5175/>。默认中文化学课把语音、观测课件与行为放在同一条原录像时间轴上。切换功能视图会保留播放器和当前时刻。

## 生成时校对

`build_review.py` 在提取概念之前生成校对稿。校对参考整段上下文、观测课件词汇和同源的其他识别结果。每次替换保留原词、建议、依据与疑点。公式、数字、否定及句式受到单独校验。模型无法完整处理的内容保留原文。

语音列表默认显示机器校对稿，点击“对照原始转写”查看原文。原始 T 段及时间不变；校对段使用 P 编号，连续校对证据使用 PB 编号。问答和概念提取使用有来源的校对证据，明确不确定的文字会从概念提取中排除。机器校对仍需回听核验。

已准备原声的课堂可增加中文二次识别：

```powershell
cd 'D:\全球校园人工智能精英大赛\4C\课堂复盘'
.venv-media/Scripts/python.exe scripts/generate_chinese_reference.py --media-dir artifacts/unified-chemistry-001-9
```

脚本使用已配置的 A6000 SSH 环境，运行 SeACo Paraformer、VAD 和标点模型。它只发布音频与视频 SHA 均一致的 `asr.secondary.json`，保留原始 Whisper 结果。依赖单独列在 `requirements-asr-chinese.txt`；服务器缓存保留模型文件与固定 revision。可用 `--host`、`--remote-python`、`--remote-root`、`--gpu` 更换运行环境。

处理另一节中文课时，可在原有流程加上 `-ChineseReference`：

```powershell
./run_lesson.ps1 -Video 'D:\课堂资料\一节课.mp4' -LessonId 'lesson-new' -Title '新的课堂' -ChineseReference
```

二次识别和校对各自保留来源。修订文件的缓存检查包含原始 ASR、课件、其他识别结果和模型信息，输入变化会重新生成校对稿。

## 查看当前状态

课堂总览优先显示当前采样画面、状态框、各状态人数和已检出参与动作比例。学生行为页提供时间趋势；原累计观测数和覆盖率放在折叠的技术诊断中。

动作模型支持阅读、书写和举手。姿态规则另列头部抬起、头部降低、手靠近头部，以及有完整腿部依据时的站姿线索。姿态规则记录实际 COCO17 关键点、置信度及阈值依据。本片未观测到的类别不填假人数。

“已检出参与动作比例”是阅读、书写和举手人数除以当前检测人数。“行为积极度参考”只对已检出的明确动作求规则均值，权重在技术详情公开。姿态本身不赋积极度分；未知与无人物画面保留未知。以上指标描述可见行为，不能证明心理投入或知识掌握。

## 实际验收

```powershell
cd '../frontend'
npm run test:review
npm run test:activity
npm run build
python scripts/verify-proofread-students.py
python scripts/verify-unified-workbench.py
```

两个浏览器脚本直接访问本机 API、原录像与本地模型。前者检查校对对照、当前状态、入口移除及手机布局；后者检查跨视图播放、独立课件导入、三类证据问答和完整导出。截图和检查记录写入 `output/playwright/`。
