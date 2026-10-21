# 音色设定 v3 · 模型对比 r0b

更新时间：2026-10-22T02:41:47+08:00（北京时间）。状态：生成成功24条、失败0条；24/24条评分已归档，等待用户下一步指示，未开始海选。

## 结果

| 模型 | 成功 | 失败 | 单条平均生成用时 | 音频总时长 |
|---|---|---|---|---|
| voxcpm2 | 8 | 0 | 18.6秒 | 102.56秒 |
| qwen3 | 8 | 0 | 37.8秒 | 111.60秒 |
| moss | 8 | 0 | 119.0秒 | 110.00秒 |

本轮为A/B/C/E四种音色，中文，seed 1/2，每个模型8条。VoxCPM2使用默认CFG 2.0、10步；Qwen3温度0.9；MOSS使用现有Backend默认参数（audio_temperature 1.5、top_p 0.6、top_k 50、repetition_penalty 1.1）。耗时来自脚本逐条输出，不含模型加载和自动测量。

24个WAV、manifest.jsonl和acoustic.json均直接复制到本目录；清单覆盖、文件解码、非静音、采样率和复制前后SHA256通过。全部24条已有声学测量。详情见[verification.json](verification.json)、[阶段状态](progress.json)、[原始生成清单](manifest.jsonl)、[原始声学结果](acoustic.json)。控制台记录见[voxcpm2-output.txt](voxcpm2-output.txt)、[qwen3-output.txt](qwen3-output.txt)、[moss-output.txt](moss-output.txt)，仅过滤加载/token进度条，原始日志保留于本机deployment目录。

自动声学阈值提示如下，仅供辅助，音色、女声感、自然度与表达由用户盲听确认：

- voxcpm2：音高偏高 8条、停顿偏多 1条
- qwen3：音高偏高 7条、停顿偏多 1条
- moss：停顿偏多 4条、音高偏高 7条、语速偏快 1条

## 部署与MOSS烟测

已拉取分支claude/nifty-ritchie-n04rar的v3代码到`2d6e4d3c2aa0d28b15b049a578442a9f07dc409f`，阅读README第0、1、1b节及DEPLOY环境C。MOSS源代码在工具包外`E:/MOSS-TTS`，版本`934d6826b084c46a0d033402174d5f8ac4ed2519`。独立环境`.venvs/xvx-moss`为Python 3.12.13；在MOSS仓库运行用户指定的editable torch-runtime安装，再安装soundfile及praat-parselmouth。torch 2.9.1+cu128、CUDA 12.8可用、算力(12,0)，实际CUDA bfloat16运算和pip check通过，未安装flash-attn。见[环境核验](moss-environment.json)、[依赖清单](requirements-xvx-moss.txt)。

ModelScope下载`openmoss/MOSS-VoiceGenerator`和`openmoss/MOSS-Audio-Tokenizer`至工具包models目录，共31个文件约11.35GB，大小全部核对通过。见[来源清单](model-sources.json)、[执行计划与源文件基线](plan.json)。

MOSS烟测成功1条、失败0条，15.76秒音频，单条生成139.2秒，整个命令158.4秒（含加载、测量、退出）。见[烟测记录](moss-smoke.json)、[音频](moss-smoke/smoke_moss_c3b284.wav)、[烟测控制台](moss-smoke-output.txt)。原smoke目录另有2条旧样本，自动测量显示3条；本次只生成MOSS的1条，不计入r0b的24条。

同一时间仅运行一个模型；每批进程退出并完成自动测量后再开始下一模型。Windows运行全部成功，无需回退。specs.py及脚本逻辑未改，common.py三个Backend也未修改；8个源文件的SHA256与拉取后的基线一致。

## 用户盲听与停止点

用户已完成本轮盲听并要求同步。已将rounds/r0b/ratings.json原样复制为本目录的[ratings.json](ratings.json)，24/24条评分与生成清单一一对应，无缺失或额外文件；复制前后SHA256为`ae245f450fc5a5165b84a47edafafd239f444512e9066b39538e2ebc6942a1ab`。同步时间与完整性核验见[ratings-sync.json](ratings-sync.json)。原始评分文件保留，本次仅同步评分与进度，不进行参数选择或启动海选。生成阶段的verification.json保留当时的核验快照，当前评分同步状态以ratings-sync.json及progress.json为准。

## 官方依据

- [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS)，2026-10-22访问、克隆，核对安装入口和模型链接。
- [MOSS-VoiceGenerator](https://modelscope.cn/models/openmoss/MOSS-VoiceGenerator)、[MOSS-Audio-Tokenizer](https://modelscope.cn/models/openmoss/MOSS-Audio-Tokenizer)，2026-10-22按官方链接及ModelScope文件接口核对仓库名并下载；项目宣传不作为本机音质验收。
