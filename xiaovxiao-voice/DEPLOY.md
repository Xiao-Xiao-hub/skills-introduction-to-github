# 部署说明（给帮忙部署的 AI / 人看）

目标：在下面这台电脑上准备 **三个独立的 Python 环境**，让本文件夹里的脚本能分别调用 VoxCPM2、Qwen3-TTS 和 MOSS-VoiceGenerator。
脚本本身不需要改，只需要把环境装好并通过最后的验收。

## 目标机器

| 项目 | 配置 |
|---|---|
| 系统 | Windows 11 |
| 显卡 | RTX 5070 Ti Laptop，12GB 显存，**Blackwell 架构（sm_120）** |
| CPU / 内存 | i9-13900HX / 32GB |

## 硬性要求

1. **NVIDIA 驱动要新**：`nvidia-smi` 右上角的 CUDA Version 需要 ≥ 12.8。
2. **PyTorch 必须 ≥ 2.7 且为 cu128 版本**。旧版本或 CPU 版在 50 系显卡上会报
   `no kernel image is available for execution on the device` / `sm_120 is not compatible`。
3. **每个模型单独一个环境**：`qwen-tts` 固定 `transformers==4.57.3`，MOSS 固定 `transformers==5.0.0`，放一起会冲突。
4. **不要装 flash-attn**（Windows 上很难编译，脚本默认不用它）。

## 环境 A：`xvx-vox`（VoxCPM2，造声 + 克隆）

```bat
conda create -n xvx-vox python=3.11 -y
conda activate xvx-vox
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install voxcpm praat-parselmouth
```

## 环境 B：`xvx-qwen`（Qwen3-TTS，造声 + 克隆）

```bat
conda create -n xvx-qwen python=3.11 -y
conda activate xvx-qwen
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -U qwen-tts praat-parselmouth
```

## 环境 C：`xvx-moss`（MOSS-VoiceGenerator，只造声）

MOSS 不是 pip 包，要先把它的代码仓库克隆到**本工具包外面**（例如 `E:\MOSS-TTS`），再在里面安装。
它的依赖里固定了 `torch==2.9.1+cu128`，支持 50 系显卡。

```bat
git clone https://github.com/OpenMOSS/MOSS-TTS.git E:\MOSS-TTS
conda create -n xvx-moss python=3.12 -y
conda activate xvx-moss
cd /d E:\MOSS-TTS
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime]"
pip install soundfile praat-parselmouth
```

装完同样检查 torch（见下一节）。不要装 flash-attn；脚本固定用 sdpa。

（`praat-parselmouth` 用来做声学测量：每轮生成完自动测音高、语调起伏、停顿等。没装也能生成，只是少了这些数字。）

（不用 conda 也行，用 `uv venv` / `python -m venv`，要求一样。）

## 每个环境装完都要检查 PyTorch 没被换掉

`pip install voxcpm` / `qwen-tts` / MOSS 有时会把 torch 换成别的版本。装完后运行：

```bat
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
```

期望输出类似：`2.x.x+cu128  12.8  True  (12, 0)`。
如果不是，重新装一次 cu128 版：

```bat
pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 模型下载

脚本第一次运行时会自动从 Hugging Face 下载：

| 用途 | 模型 | 用在哪 |
|---|---|---|
| VoxCPM2 造声和克隆 | `openbmb/VoxCPM2` | 环境 A，所有步骤 |
| Qwen3 造声 | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 环境 B，step1、step2 |
| Qwen3 克隆（可选） | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 环境 B，只有 step3 用 `--model qwen3` 时需要 |
| MOSS 造声 | `OpenMOSS-Team/MOSS-VoiceGenerator` | 环境 C，step1、step2 |
| MOSS 音频编解码器 | `OpenMOSS-Team/MOSS-Audio-Tokenizer` | 环境 C，和上面一起用（加载时自动下载） |

国内网络下载慢时，二选一：

- 运行脚本前先 `set HF_ENDPOINT=https://hf-mirror.com`
- 或用 ModelScope 下到本地，再给脚本加 `--model-path 本地路径`：
  ```bat
  pip install modelscope
  modelscope download --model OpenBMB/VoxCPM2 --local_dir D:\models\VoxCPM2
  modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local_dir D:\models\Qwen3-TTS-VoiceDesign
  modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --local_dir D:\models\Qwen3-TTS-Base
  modelscope download --model openmoss/MOSS-VoiceGenerator --local_dir D:\models\MOSS-VoiceGenerator
  modelscope download --model openmoss/MOSS-Audio-Tokenizer --local_dir D:\models\MOSS-Audio-Tokenizer
  ```
  MOSS 用本地模型时要同时给两个路径：`--model-path D:\models\MOSS-VoiceGenerator --codec-path D:\models\MOSS-Audio-Tokenizer`。
  （ModelScope 上的仓库名以 [MOSS-TTS 的 ModelScope 合集](https://www.modelscope.cn/collections/openmoss/MOSS-TTS) 为准。）

## 验收（每个环境一条，都要通过）

在本文件夹里：

```bat
conda activate xvx-vox
python step1_audition.py --model voxcpm2 --smoke

conda activate xvx-qwen
python step1_audition.py --model qwen3 --smoke

conda activate xvx-moss
python step1_audition.py --model moss --smoke
```

每条命令最后应显示「成功 1 条，失败 0 条」，并在 `rounds\smoke\` 里生成一个能正常播放的 wav（一个女声读试音台词）。
验收通过后可以删掉 `rounds\smoke` 文件夹。

## 常见问题

- **显存不足**：同一时间只跑一个模型；关掉占显存的游戏、浏览器硬件加速等。
- **VoxCPM2 打印 `torch.compile disabled`**：正常，Windows 上缺 triton，会自动用不编译的模式，只是慢一点。脚本默认就不编译。
- **`SoX could not be found`**：qwen-tts 的提示，不影响本工具包用到的功能。
- **报错和 torchcodec / FFmpeg DLL 有关**：装一份共享库版 FFmpeg，例如 conda 环境里 `conda install -c conda-forge "ffmpeg<8"`。
- **MOSS 加载时报 cuDNN / SDPA 相关错误**：脚本已按官方示例关掉 cuDNN SDPA；仍报错时把报错原文记下来，先别改脚本。
- **包更新后接口变了**：只需要改 `common.py` 里的 `VoxCPM2Backend` / `Qwen3Backend` / `MossBackend` 三个类，其它脚本不用动。
  当前对应的接口：VoxCPM2 `VoxCPM.from_pretrained(...).generate(text, prompt_wav_path, prompt_text, reference_wav_path, cfg_value, inference_timesteps[, seed])`（voxcpm 2.0.3 没有 seed 参数，脚本会自动判断）；
  Qwen3 `Qwen3TTSModel.from_pretrained(...).generate_voice_design(text, instruct, language)` / `create_voice_clone_prompt` / `generate_voice_clone`；
  MOSS `AutoProcessor`/`AutoModel.from_pretrained(..., trust_remote_code=True)`，`processor.build_user_message(text, instruction)` → `model.generate(..., audio_temperature, audio_top_p, audio_top_k, audio_repetition_penalty)` → `processor.decode`。
