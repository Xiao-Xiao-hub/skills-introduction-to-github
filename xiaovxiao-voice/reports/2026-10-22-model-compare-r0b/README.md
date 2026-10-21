# 音色设定 v3 · 模型对比 r0b

更新时间：2026-10-22T02:17:01+08:00（北京时间）。状态：VoxCPM2 8/8、Qwen3 8/8、MOSS 4/8，总进度20/24。当前阶段：moss_running。

已拉取分支 claude/nifty-ritchie-n04rar 到 `2d6e4d3c2aa0d28b15b049a578442a9f07dc409f`，阅读 README 第0、1、1b节及 DEPLOY.md 环境C。MOSS源代码单独克隆到 E:/MOSS-TTS，版本为 `934d6826b084c46a0d033402174d5f8ac4ed2519`；独立环境为工具包内 .venvs/xvx-moss，Python 3.12.13。

## 本轮计划

1. 安装 MOSS 的 torch-runtime，再安装 soundfile、praat-parselmouth；验证 cu128/CUDA及算力，不安装flash-attn。
2. 从ModelScope获取 openmoss/MOSS-VoiceGenerator 和 openmoss/MOSS-Audio-Tokenizer；仓库名已通过官方仓库链接与ModelScope文件接口核实，两套共约11.35GB。
3. MOSS真实烟测1条，记录成功/失败与单条生成用时。
4. 依次运行 VoxCPM2、Qwen3、MOSS 的 r0b，每个模型 A/B/C/E × 中文 × seed 1/2，共8条，合计24条。Vox使用默认CFG 2/10步，Qwen温度0.9，MOSS使用脚本已有的官方推荐参数。
5. 复制全部WAV、manifest.jsonl、acoustic.json到本目录，核对后提交推送。

如果MOSS在Windows无法安装或运行，记录原始报错并先完成另外两个模型的16条；不会把未生成的MOSS音频写成成功。本轮不改specs.py或脚本逻辑，若确有接口问题，只允许调整common.py中的三个Backend类并记录差异。同一时间仅运行一个模型。

计划和源文件基线见 [plan.json](plan.json)，模型来源见 [model-sources.json](model-sources.json)，阶段状态见 [progress.json](progress.json)。

## 停止点

完成后只进行用户盲听评分。用Edge打开工具包内rater.html，点「打开文件夹（自动保存）」，选择rounds/r0b。用户说「评完了」后，再把ratings.json复制到本目录并提交推送；当前不复制评分，不开始海选。

## 官方依据

- [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS)，访问及克隆日期2026-10-22；用于安装入口和模型链接核对。
- [MOSS-VoiceGenerator](https://modelscope.cn/models/openmoss/MOSS-VoiceGenerator)、[MOSS-Audio-Tokenizer](https://modelscope.cn/models/openmoss/MOSS-Audio-Tokenizer)，访问及文件清单查询日期2026-10-22；用于获取模型代码与权重，不把项目自测宣传当作本机音质验收。

## 环境安装检查点

MOSS环境已成功安装，torch2.9.1+cu128、CUDA12.8可用、算力(12,0)，实际CUDA bfloat16运算和pip check通过，未安装flash-attn。详情见 [moss-environment.json](moss-environment.json)，版本清单见 [requirements-xvx-moss.txt](requirements-xvx-moss.txt)。MOSS实际生成烟测也已通过，见下节。

## MOSS烟测通过

成功1条、失败0条，15.76秒音频，单条生成用时139.2秒；整个命令进程158.4秒（包含加载、自动测量和退出等）。没有接口修改。见 [moss-smoke.json](moss-smoke.json)、[烟测音频与清单](moss-smoke/)、[非进度条控制台输出](moss-smoke-output.txt)。控制台文件仅去掉权重加载和token进度条，完整原始日志保留在本机deployment目录。原smoke目录已有2条旧样本，因此自动测量显示共3条；本次烟测仅运行MOSS的1条。
