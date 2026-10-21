# 音色设定 v2 · 参数校准 r0

更新时间：2026-10-22T01:15:02+08:00（北京时间）。状态：24条评分已原样归档，评分分析完成；等待用户决定下一步，正式海选未启动。

## 结果

| 模型 | 参数组合 | 成功 | 失败 | 平均单条生成用时 |
|---|---|---|---|---|
| VoxCPM2 | A/B/D/F × 中文 × seed 1 × CFG 1.5/2.0 × 步数 10/20 | 16 | 0 | 48.6 秒 |
| Qwen3 | A/B/D/F × 中文 × seed 1 × temperature 0.9/1.0 | 8 | 0 | 36.8 秒 |
| 合计 | 校准 r0 | 24 | 0 | 不含模型加载时间 |

两套模型依次运行，VoxCPM2 进程退出后才启动 Qwen3。生成输出保留在工具包内 `rounds/r0`，全部 24 个 WAV、manifest.jsonl、acoustic.json 已复制到本报告目录；WAV 副本逐一核对 SHA256 一致，未裁剪、归一化或重新编码。总音频大小为 29,304,096 字节。

## 文件与证据

- [manifest.jsonl](manifest.jsonl)：24 条音频的描述、台词、种子、参数和文件名；WAV 位于本目录，保持清单的相对路径。
- [acoustic.json](acoustic.json)：24 条声学测量结果。
- [ratings.json](ratings.json)：用户完成的24条原始评分，按原文件字节复制。
- [ratings-archive.json](ratings-archive.json)：复制时间、评分更新时间与文件 SHA256。
- [ratings-analysis.md](ratings-analysis.md)：模型、参数、方向、标签及备注分析。
- [ratings-summary.json](ratings-summary.json)：完整分组统计与配对比较。
- [measure-output.txt](measure-output.txt)：用户指定的 measure.py 命令完整控制台输出，共 25 行，未做摘要替换。
- [verification.json](verification.json)：数量、参数组合、音频 SHA256、生成用时、退出码和源文件完整性检查。
- [environment.json](environment.json)：两个环境安装 praat-parselmouth 后的 torch/CUDA 检查。
- [plan.json](plan.json)：本轮参数计划、启动时间和代码基线。
- [generation-progress.json](generation-progress.json)：最终阶段状态，保留等待评分和参数选择的边界。

## 执行与检查

已拉取分支 `claude/nifty-ritchie-n04rar` 到 `a37e7072c74b962028b34ee2f603c3913b8b70f8`，完整阅读 README 第 1、2、3 节。两环境均安装 praat-parselmouth 0.4.7；安装后 torch 均为 2.11.0+cu128，CUDA 12.8 可用，算力均为 (12, 0)。

实际运行的命令如下，均从工具包目录执行：

```powershell
.\.venvs\xvx-vox\Scripts\python.exe step1_audition.py --model voxcpm2 --model-path .\models\VoxCPM2 --only A,B,D,F --langs zh --seeds 1 --cfg 1.5,2.0 --steps 10,20 --out rounds\r0
.\.venvs\xvx-qwen\Scripts\python.exe step1_audition.py --model qwen3 --model-path .\models\Qwen3-TTS-VoiceDesign --only A,B,D,F --langs zh --seeds 1 --temperature 0.9,1.0 --out rounds\r0
.\.venvs\xvx-vox\Scripts\python.exe measure.py rounds\r0
```

Vox 自动测量 16 条，Qwen 结束后自动补测 8 条。因此最后显式运行 measure.py 时显示「新测 0 条，共 24 条」，复用已有结果。完整原文见 measure-output.txt；程序提示停顿偏多 24 条、气声偏重 2 条、语速偏慢 1 条。这些是脚本提示项，不是用户评分或自动选参结论。

对基线的 8 个源文件逐一核对 SHA256，全部未改，包括 specs.py、common.py、measure.py、step1–step4 和 rater.html；本轮没有 Backend 接口修改。manifest 的模型数、A/B/D/F、中文描述、seed 1 及各参数组合均与本轮要求一致，acoustic.json 覆盖全部 24 条。

## 停止点与下一步

用户已明确说「评完了」，24/24条评分已完成，ratings.json 已从 rounds/r0 原样复制到本报告目录。归档副本与本机原件逐字节、SHA256 一致；原评分文件保留在运行目录。

评分分析已完成，见 [ratings-analysis.md](ratings-analysis.md)。评分分布为2分14条、3分9条、4分1条，尚无5分。仅将Vox的CFG 2/10步、Qwen的温度0.9列作候选起点；最终参数未选定，暂不启动120条海选，等待用户的新指令。
