# 独立 CPU 环境

参考平台为 Python 3.11、Windows/Linux x64。需要 Git；无需 CUDA。技能源码不附模型、音频和第三方包，安装资源写入指定 runtime 目录，不修改全局 Python。

## 新机器

在 skill 目录运行，先看计划，再执行安装：

```powershell
py -3.11 scripts/setup_runtime.py --runtime-root D:/BgmRuntime
py -3.11 scripts/setup_runtime.py --runtime-root D:/BgmRuntime --apply
D:/BgmRuntime/.venv/Scripts/python.exe scripts/understand_bgm.py --tools-dir D:/BgmRuntime --audio "输入.mp3" --output-dir "新输出目录"
```

Linux 使用 `python3.11` 和 `<runtime>/.venv/bin/python`。也可设置 `BGM_ANALYSIS_ROOT` 指向 runtime；之后仍用该 runtime 的私有 Python 调用 skill。

`--apply` 安装 CPU PyTorch/torchaudio 和 `requirements-runtime.txt`，检出 `runtime.lock.json` 中的源代码 commit，下载并验证模型的 SHA256/字节数。模型及元数据约142MB；Python、CPU推理依赖和 pip 缓存另占空间。安装阶段需要联网，分析阶段使用本地缓存。Whisper/SenseVoice 不在默认安装范围。

安装器遇到现有文件或源代码与锁不符时停止，保留原文件；不会自动升级、覆盖或清理旧环境。中断下载留作 `.install-part`，核查后只处理该文件再重试。官方资源不可达或内容变更会明确失败，不能跳过hash核验来宣称安装成功。

## 检查已有环境

```powershell
py -3.11 scripts/doctor_runtime.py --runtime-root "已有runtime目录"
py -3.11 scripts/doctor_runtime.py --runtime-root "已有runtime目录" --check-imports
```

默认核对26个文件身份、4个源代码版本和依赖是否可发现；`--check-imports` 额外在子进程验证主要 Python/native 依赖导入，不加载权重、不推理。必须用实际将运行分析的 Python 调用 doctor。`ready=true` 是环境检查，不代表音乐结果准确；新机器仍需运行一段实际音频。

核心信号环境可用 `setup_runtime.py --profile signal`，但默认技能还要求段落/语义环境；仅信号运行需显式 `--structure-mode off --semantics-mode off`，输出如实为部分分析。

## 固定来源及适用范围

- Beat This `small0`：CPJKU 官方checkpoint；Sonara和Beat This适配脚本已包含在 `scripts/`，不再依赖runtime根目录的临时脚本。
- All-In-One CPU移植：openmirlab `all-in-one-infer` / `madmom-infer` / `demucs-infer`，固定commit见锁文件；实际采用原作者Harmonix checkpoint和HTDemucs分轨。未宣称与原版NATTEN逐数值等价。
- 声音语义：w-hc PyTorch YAMNet固定commit/权重；Essentia Discogs-Effnet和分类头按发布文件锁定。若镜像内容变化，hash核验会失败。
- 第三方权重有独立授权；特别是Harmonix及部分Essentia权重标为CC BY-NC-SA 4.0，不能因项目源码公开就认为所有模型可任意商用。授权说明以锁文件、源仓库和模型卡为准。

验证边界：已部署的Windows CPU环境运行过真实音频；安装器可在已有环境执行计划、静态核验和导入检查。完整依赖安装命令和Linux流程仍需在干净机器持续验证，不能把已有缓存成功等同于冷启动安装成功。
