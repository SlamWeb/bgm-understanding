# 声源与情绪模型

`analyze_semantics.py` 复用本机 CPU PyTorch 与局部 `semantic-packages/onnxruntime`。默认运行不下载文件。权重位于运行时根目录的 `models/semantic-heads/`、`models/wav2small/`，YAMNet 复用 `panns-test/yamnet.pth`。

| 模型 | 用途 | 权重大小 | 来源 |
|---|---|---:|---|
| YAMNet | 明确的 speech/singing 类候选 | 15.08MB，复用 | [Google 模型与前端](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)、[现有 PyTorch 移植](https://github.com/w-hc/torch_audioset) |
| Discogs-Effnet | 共享音乐 embedding | 18.03MB | [MTG 官方模型](https://essentia.upf.edu/models/feature-extractors/discogs-effnet/) |
| MTG-Jamendo instrument | 40 类乐器与制作声音标签 | 2.71MB | [官方模型与标签元数据](https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/) |
| 四个 mood 头 | happy/sad/relaxed/aggressive 独立音乐情绪候选 | 共2.06MB | [官方模型目录](https://essentia.upf.edu/models/classification-heads/) |
| voice/instrumental 头 | 音乐中有人声的候选；不能区分说话与唱歌 | 0.514MB | [官方元数据](https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.json) |
| audEERING Wav2Small2.0 | 语音激动程度、支配感、正负情感三个连续属性 | 68,716bytes，约17K参数 | [发布者模型卡与权重](https://huggingface.co/audeering/wav2small) |

MTG 与 audEERING 这些权重采用 CC BY-NC-SA 4.0，商业许可另询发布者。官方站点本机连接失败时，音乐模型从 Hugging Face 的 `rlupi/deep-cuts-models-noncommercial` 固定文件下载；乐器头来自 `coiluck/kokone-music-for-mobile` 的固定 commit `9cfeb133ba8ea25385a239834395c3a28645a93c`。每次报告保留实际文件 SHA256。乐器头为 `9ae2d9e763d66bd8eed654d1ac3aa171e6539cb8a0e11f3dcd53df1428980802`，Wav2Small 为 `86ccef83ce942ff34299a44684e9b69bf37021dd4b69325c0dcb3decb0df2f4d`。

## 分数与适用域

- 两个人声模型使用不同任务与分数尺度。`vocal_score=max(...)` 只作宽松 ASR 路由；原始 `yamnet_vocal_score`、`music_voice_score` 独立保留。YAMNet 低而音乐 voice 头高时，类型为 `uncertain_vocal_candidate`，记录分歧。文本仍需实际声学转写证据。分离轨的能量可能含残留，不能独自确认人声。
- 音乐标签与四个 mood 分数属于独立多标签/二分类模型输出，不要求和为1，不是已校准概率。纯朗读输入的音乐标签推荐清空，原始值保留并标 `low_spoken_audio`。
- `emotion.speech_emotion` 仅在明确 speech 类证据覆盖足够音频窗时运行 Wav2Small。唱歌或类型不明的人声不自动进入这个语音情绪分支。输出 arousal/dominance/valence 回归值，不硬转为“悲伤”“释然”等叙事解释，也不代表说话者心理状态。
- Wav2Small 论文以 MSP-Podcast 语音为评估来源；本次17K发布模型卡没有建立多语种验证，其中文朗读迁移效果尚未人工验证。保留这个边界，避免把一次纯数值输出当作跨语言准确性验证。

## 已完成的机械验证

Google 官方 NumPy `mel_features` 按 YAMNet 参数对照：log-mel 最大绝对误差 `5.43e-7`。官方 Essentia WASM `TensorflowInputMusiCNN` 在静音、440Hz 正弦、固定随机噪声三个512采样帧上对照：96维特征最大绝对误差 `1.895e-5`。这些检查验证特征实现，不证明乐器/情绪标签符合人的听感。现有第三方 PyTorch YAMNet 与官方 TensorFlow 完整网络逐数值等价仍未验证。

Wav2Small 的卷积频谱核与 mel 矩阵直接来自发布者 safetensors；`speech_emotion.py` 按模型卡实现小 CNN，严格加载所有 tensor，使用已有 PyTorch，不安装 Transformers/librosa，不加载可执行远端代码。
