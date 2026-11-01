# 运行与资源

新机器先看 [installation.md](installation.md)：固定依赖、upstream commit、模型hash和私有CPU环境均由现有安装脚本管理。分析时不自动下载权重或修改依赖。`doctor_runtime.py` 可检查环境。

现有本机 runtime 为 `D:/SelfMedia/管理/工具/bgm-analysis`，使用Python3.11；其他机器用 `--tools-dir <runtime>` 或 `BGM_ANALYSIS_ROOT`。结构运行时默认 `<runtime>/all-in-one-trial`，可用 `--structure-runtime`覆盖。模型目录结构由 `runtime.lock.json` 定义；技能内已包含Sonara和Beat This适配脚本。

默认完整分析不超过600秒，超长明确报错。需要选段时用 `--start` / `--duration` 并保留偏移，跨段结构上下文可能损失。每次使用新输出目录，保留失败日志和原音频。

默认4线程，每阶段进程树RSS上限2200MB、1800秒。低内存用 `--threads 2`，音源串行处理。结构和语义失败仍输出 `status: partial` 并注明缺项；信号和分频失败不生成完成交付。模型文件大小不是运行峰值内存。没有后台模型服务。

默认关闭Whisper/SenseVoice，专注BPM、拍点、起音、走势和段落。只在用户明确重新需要文字时查看 [transcript-review.md](transcript-review.md)；已有可选脚本需另行部署其模型。All-In-One段落分析本身使用HTDemucs，关闭歌词不会移除这部分计算。

仅需信号时可用 `--structure-mode off --semantics-mode off`；结果是部分分析，不能称完整All-In-One。结构模型默认八fold `harmonix-all`；`--structure-model harmonix-fold0`降低计算量。语义、voice和情绪模型边界见 [semantic-models.md](semantic-models.md)。首次9/28秒单fold含分离测试约57/105秒、峰值约930/975MB，非所有音频的性能承诺。

`--reuse-core <旧输出目录>` 复用同音源raw/features/structure，在新目录重新生成后续结果。复用会核对输入hash、区间、WAV与报告hash；整首复用要求原receipt的 `requestedSelection=whole_file`。旧receipt或选段须显式给出相同duration。已完成分离而结构后续中断时，用 `--reuse-structure-run <缓存run目录>` 核对原WAV与四个stem后恢复，不重做分离。结构特征在独立子进程内逐stem转换为普通NumPy数组并释放FFT中间量，以控制长曲内存。

交付后用 `validate_agent_json.py <主JSON> --require-enriched --decode-source`核验原音频完整覆盖、PCM、证据hash及时间点。它验证机械一致性，不等于人工听感验证。
