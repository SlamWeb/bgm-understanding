---
name: bgm-understanding
description: 输入本地音乐或音频，生成供创作和剪辑 agent 使用的 BGM 理解 JSON，提供 BPM、完整拍点、起音强度、声音走势和段落变化；默认不识别歌词。
---

# MP3 → BGM 理解文件

给定本地音频，运行已有脚本，交付一份 `bgm-understanding.json`。不要求用户先标时间，也不需要素材集或选题；下游 agent 再结合它们决定镜头、生图、切换和转场。

```powershell
py -3.11 "<本 skill 目录>/scripts/understand_bgm.py" --audio "<音频绝对路径>" --output-dir "<新的输出目录>"
```

首次安装或迁移机器时读 [references/installation.md](references/installation.md)，使用内置安装计划与运行环境检查。已部署的机器直接运行入口；模型和音频不放进 skill 仓库。

1. 运行单入口。默认完整分析不超过 600 秒的音频，依次执行节拍/起音、能量与分频、All-In-One、声音标签/音乐情绪。默认关闭 Whisper 和 SenseVoice 转写，重点供剪辑 agent 依据音乐组织素材；已有 ASR 代码作为可选能力保留。各模型串行运行并释放，使用 CPU、线程与进程树内存监督。本机依赖、资源调整与故障恢复见 [references/runtime.md](references/runtime.md)。
2. 默认交付 `bgm-understanding.json`（schema `bgm-editing-brief-v2`），节奏/逐拍时间、小节起拍、起音及强度、段落/走势都在主文件。以信息是否支持剪辑判断为准，不以固定行数删减必要数据。入口自动校验音源、时间覆盖、完整卡点及描述开销，校验结果保存在 `final-validation.json` / `output-quality.json`；失败会报错。模型逐窗分数和诊断另存 `bgm-analysis-detail.json`，只向下游交付主文件路径；机械通过不代表听感正确。
3. `overview` 给BPM/音色概括；`timeline` 给段落、能量走势、声音密度及乐器/情绪候选；`change_candidates` 给变化时间及依据；`sound_motion` 给最多48个声音走势窗口（能量、低中高频占比与起音密度）；`cut_points` 直接给全部节拍、小节首拍和检测到的起音。默认 `text_status=skipped_not_requested`，`text_cues=[]`，表示未请求转写。无需另写脚本找点；仅需筛选区间时可用 `extract_bgm_events.py --detail <主JSON> --start <秒> --end <秒>`。根据素材阅读时间和叙事推进选择点，不机械每拍切图。字段见 [references/output.md](references/output.md)。

只有用户重新明确需要文字时，才用 `--lyrics-mode full` 开启转写，并读 [references/transcript-review.md](references/transcript-review.md)；额外语音复核须显式 `--speech-mode auto`。默认不为歌词为空重试或升级模型。乐器、人声活动与音乐情绪候选的适用域见 [references/semantic-models.md](references/semantic-models.md)。

所有时间相对于本次解码音频开头；转回原文件加 `audio.source_start_seconds`，裁切或变速后另作映射。人工标记是可选的创作偏好，若有则另存，不冒充自动模型结果。先完成一次自动输出，再考虑是否需要人修正。

段落、乐器、人声活动和情绪是模型候选。乐器窗口不能确定窗口内每个起音的声源，音乐情绪标签不能替代听众感受；模型分数是来源各异的未校准分数。已有样本的机械验证与人工听感验证分开记录。
