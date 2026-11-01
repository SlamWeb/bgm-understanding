# BGM Understanding

**一首音频 → 一份 Agent 能直接用于剪辑的音乐时间图。**

提供 BPM、完整拍点与小节首拍、起音及强度、声音起伏、段落变化候选。下游 Agent 结合选题和素材，决定在哪切、怎样转场、何时铺垫或加速。

默认 CPU、串行处理，不识别歌词。主 JSON 保留剪辑信息，逐窗模型分数和运行诊断另存。

## 使用

需要 Python 3.11。先按 [安装说明](references/installation.md) 准备独立运行环境和模型，再运行：

```powershell
python scripts/understand_bgm.py --audio "music.mp3" --output-dir "analysis"
```

使用安装环境的 Python 执行；每次选择新输出目录。把 `analysis/bgm-understanding.json` 交给创作或剪辑 Agent。通过 Codex 使用时安装整个目录为 `bgm-understanding` skill。

## 输出

| 字段 | 解决什么问题 |
| --- | --- |
| `overview` | 大致多快、主要是什么声音 |
| `cut_points` | 每拍、小节起拍、具体起音在哪里 |
| `timeline` / `change_candidates` | 哪一段进入了另一种状态 |
| `sound_motion` | 哪些时间变响、变密、低频增多 |

[字段说明](references/output.md) · [真实输出示例](examples/preview-bgm.json) · [回归验证](references/release-validation.md)

主输出有结构和描述开销检查，完整拍点不会因精简被删掉。段落、乐器和情绪是模型候选；起音不等于已识别的鼓或琴键。检测结果不是固定剪辑方案，也没有保证识别“副歌”或“最爽时刻”。

## 开发与模型

```powershell
python -m unittest discover -s scripts -p "test_*.py"
```

仓库不包含音乐、模型权重、缓存或分析原件。安装器有固定版本和模型身份校验；[模型来源及适用边界](references/semantic-models.md) 与第三方许可证单独适用，部分模型限非商业用途。本项目代码的授权不替代模型或音乐的授权。
