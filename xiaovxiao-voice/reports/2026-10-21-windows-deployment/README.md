# Windows 部署与双模型烟测报告

日期：2026-10-21（北京时间）。本轮交付：两个独立的 Python 3.11 环境、两条真实生成的试听 WAV，以及可核对的运行记录。场景为在 RTX 5070 Ti Laptop 12GB 上寻找 AI 配音音色。

部署文档已完整阅读，使用仓库现有脚本。部署代码基线为 `42288cbee80388ff3e26e89d52e9e3790b497807`，分支为 `claude/nifty-ritchie-n04rar`。

## 结果与样本

| 模型 | 最终烟测 | WAV | 采样率 | 时长 | 单条生成用时 |
|---|---|---|---|---|---|
| VoxCPM2 | 成功 1 条，失败 0 条 | [smoke_vox_5b280b.wav](smoke/smoke_vox_5b280b.wav) | 48,000 Hz | 13.28 秒 | 35.0 秒 |
| Qwen3-TTS VoiceDesign | 成功 1 条，失败 0 条 | [smoke_qwen_7020d0.wav](smoke/smoke_qwen_7020d0.wav) | 24,000 Hz | 17.12 秒 | 64.9 秒 |

生成用时来自脚本单条输出，不含模型加载和下载时间。两次测试依次进行，Qwen 进程退出后才加载 VoxCPM2。

- [原始样本清单](smoke/manifest.jsonl)：保留描述、台词、种子、生成时间及模型信息，音频文件名使用相对路径。
- [结构化验收证据](evidence.json)：环境版本、最终结果、音频波形指标及 SHA256。
- WAV 均为单声道 PCM 16-bit，读取成功、采样值有限且含非静音信号；仓库中的副本与本机生成文件 SHA256 一致，未做归一化、裁剪或重新编码。
- **尚未完成用户人工试听。是否正常可听、是否女声及音色是否满意，均等待用户确认。正式海选未启动，当前为 0/120。**

## 环境与下载

工具包本机目录：`E:\skills-introduction-to-github\xiaovxiao-voice`。两个环境分别为 `.venvs/xvx-vox`、`.venvs/xvx-qwen`，Python 均为 3.11.15。

| 检查项 | xvx-vox | xvx-qwen |
|---|---|---|
| torch / torchaudio | 2.11.0+cu128 / 2.11.0+cu128 | 2.11.0+cu128 / 2.11.0+cu128 |
| CUDA 版本 / 可用 | 12.8 / True | 12.8 / True |
| GPU 算力 | (12, 0) | (12, 0) |
| 实际 CUDA bfloat16 矩阵运算 | 通过 | 通过 |
| 模型包 | voxcpm 2.0.3；未安装 qwen-tts | qwen-tts 0.1.1；未安装 voxcpm |
| transformers | 5.20.2 | 4.57.3 |
| flash-attn | 未安装 | 未安装 |
| pip check | No broken requirements found. | No broken requirements found. |

NVIDIA 驱动为 616.92，显卡实际查询结果为 NVIDIA GeForce RTX 5070 Ti Laptop GPU，显存 12,227 MiB，驱动显示 CUDA UMD 13.4。两个环境均先从 `https://download.pytorch.org/whl/cu128` 安装 torch、torchaudio，再安装各自模型包；之后重新验证 torch 版本和 CUDA，未被替换为 CPU 或旧版。

环境完整版本清单：[xvx-vox](requirements-xvx-vox.txt)、[xvx-qwen](requirements-xvx-qwen.txt)。这两份清单属于不同环境；重建时仍须先按 DEPLOY.md 安装 cu128，不能把两份清单合并安装。

模型从 ModelScope 下载到本地：

| 模型来源 | 工具包内本机路径 | 文件检查 |
|---|---|---|
| [OpenBMB/VoxCPM2](https://modelscope.cn/models/OpenBMB/VoxCPM2) | `models/VoxCPM2` | 9 个文件，合计 4,960,732,453 字节；逐文件大小核对通过 |
| [Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) | `models/Qwen3-TTS-VoiceDesign` | 14 个文件，合计 4,520,164,607 字节；逐文件大小核对通过 |

来源访问和下载日期均为 2026-10-21，来源用途仅为获取此次实际运行的权重。未下载当前不需要的 Qwen3 Base 克隆模型；权重、虚拟环境及下载缓存未加入 Git。

## 实际烟测命令

以下命令均在工具包目录运行，使用已下载的本地模型，未开启 flash-attn 或 torch.compile：

```powershell
.\.venvs\xvx-qwen\Scripts\python.exe step1_audition.py --model qwen3 --smoke --model-path .\models\Qwen3-TTS-VoiceDesign
.\.venvs\xvx-vox\Scripts\python.exe step1_audition.py --model voxcpm2 --smoke --model-path .\models\VoxCPM2
```

最终关键输出：

```text
[1/1] smoke_qwen_7020d0  A软糯忧郁 [zh]  17.12s 音频，用时 64.9s，预计还要 0.0 分钟
完成：成功 1 条，失败 0 条。

[1/1] smoke_vox_5b280b  A软糯忧郁 [zh]  13.28s 音频，用时 35.0s，预计还要 0.0 分钟
完成：成功 1 条，失败 0 条。
```

## 兼容修复与检查范围

VoxCPM2 第一次生成失败 1 条，报错为 `VoxCPM._generate() got an unexpected keyword argument 'seed'`。检查已安装的 voxcpm 2.0.3 后，确认其 generate 接口不接受 seed 参数。

只修改了 [common.py 的 VoxCPM2Backend](../../common.py)：生成前调用已有的 `seed_everything(seed)`，移除传给 generate 的 seed 参数。描述、台词、CFG、推理步数、种子数和任务调度逻辑均未改。重新运行同一条 VoxCPM2 烟测命令后成功。

检查证据：

- 模型包安装后的 torch、CUDA、算力与实际 GPU 运算均通过；两个环境分别运行 pip check 通过。
- 对部署前后的源文件做 SHA256 比较：只有 common.py 改变；specs.py、step1–step4 和 rater.html 均未改变。
- AST 对比确认 common.py 中 VoxCPM2Backend 之外的代码未改，Qwen3Backend 未改。
- git diff --check 通过；此次同步仅增加记录、样本与 README 导航，没有重新运行已经通过的模型烟测。
- Qwen 日志中缺少 flash-attn、SoX 的提示未阻止本次生成，按 DEPLOY.md 保持不安装 flash-attn。

## 下一步与同步

等待用户试听两条样本，明确确认有声音、是女声后，再依次运行两套环境的正式海选，各 60 条，共 120 条，输出到 `rounds/r1`。中断后重新运行同一命令续跑。运行期间报告实际进度和估算剩余时间，并按 [PROGRESS.md](../../PROGRESS.md) 同步实质进展与结果。

正式海选完成后，再用 Edge 打开 `rater.html`，点「打开文件夹（自动保存）」并选择 `rounds/r1`，开始盲听打分。当前报告不代表人工试听或正式海选已验收。
