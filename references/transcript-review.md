# 自动转写后的 Agent 复核

本页的转写复核和 `enrich_understanding.py` 操作针对详细分析文件。修改明细后，再用 `summarize_bgm.py --analysis <新明细> --output <新简版>` 刷新默认交付；旧简版的SHA会因明细改变而失效。

`analyze_lyrics.py` 先用独立声源候选决定转写窗口，再按至多20秒核心窗解码；窗口有上下文，词按中点归属核心区，时间回到完整 WAV。`coverage` 表示整次请求，`analysis_windows` 表示实际解码区间。未保留文字代表没有获支持的转写，不是绝对无歌声。音乐 voice 头与 YAMNet 分歧时，`uncertain_vocal_candidate` 是宽松路由，不是确认人声。

默认保留 Whisper base（74M参数）的原始短语、词时轴和诊断。强讲话音频另用 `analyze_speech_semantics.py` 的 SenseVoice INT8 提供独立声学文本及语气标签；两者都可能错同音字。音频知识与文件名可帮助理解语境，但不能替代实际声学结果。

升级模型前核对语言。歌声混音中，单个声源分数最高的窗口可能实际是器乐桥段，语言检测也可能把英语错判为拉丁语；机械完整和句子看似流畅不能排除此错误。已有同源完整转写和多个实际声乐片段一致支持某语言时，单入口用 `--asr-language en`（按证据替换代码），独立转写脚本用 `--language en`。这是语言条件，不是预塞目标歌词。新模型语言与既有声学证据冲突时先检查短片段；错误语言的checkpoint保留作失败证据，不能混进最终候选或按正常恢复。

若已识别文字明显形成一个引用或高确定性的同音字错误，可增加上下文纠错候选。原 `text` 与 `words` 保留不变；纠错后的汉字没有重新做逐字音频对齐。将候选另存文件，再用 `enrich_understanding.py --core <core-understanding.json> --semantics <semantics.json> --lyrics <lyrics.json> --corrections <corrections.json> --speech <可选完整speech-semantics.json> --output <新的最终JSON>` 合并。

纠错文件采用以下字段，使用真实报告的音源与sha；每条 start/end/raw_text 必须准确匹配原ASR短语：

```json
{
  "audio": {"source": "canonical WAV绝对路径", "sha256": "真实SHA", "duration_seconds": 15.6, "time_origin": "canonical_wav_start"},
  "raw_lyrics_sha256": "lyrics.json的真实SHA",
  "basis": "纠错的语言上下文依据",
  "segments": [{"start": 0.0, "end": 1.18, "raw_text": "原ASR文字", "normalized_text_candidate": "纠错候选"}]
}
```

混音中某个疑似说话短句需要复核时，可运行 `analyze_speech_semantics.py --wav <canonical.wav> --output <新报告.json> --runtime-root <bgm-analysis目录> --threads 1 --start <秒> --duration <秒>`。选段报告保留完整 WAV 的身份，但 coverage 明确为选段，不冒充整曲识别。将其作为 `lyrics.verification` 或原始证据引用，完整报告才能用 `enrich_understanding.py --speech` 合并。

本次三曲测试中的“伴奏”文件实际出现两个声学模型都识别到的英文制作人广告候选；不能依据文件名把全部文本强制清空，也不能把分轨残留或重复解码直接当歌词。此例说明一种处理思路，未来由输入与识别证据决定。
