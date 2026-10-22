# 晓念 V10 旁白试配

更新时间：2026-10-22T10:32:40+08:00（北京时间）。状态：16条试配全部生成并测量，成功16条、失败0条；已归档，等待用户盲听指定正式配音音色。

## 试配结果

| 参考音色 | 成功 | 失败 | 单条平均生成用时 |
|---|---|---|---|
| r0b_vox_821713 | 4 | 0 | 34.5秒 |
| r0b_vox_49a14f | 4 | 0 | 16.9秒 |
| r0b_moss_611767 | 4 | 0 | 21.4秒 |
| r0_vox_959950 | 4 | 0 | 28.3秒 |

统一使用VoxCPM2，4个参考音色 × 4组（05-06、07、14、23a），style方式、seed 1、CFG 2.0、10步，共16条。MOSS来源候选只用作参考音频，本轮没有运行MOSS模型。命令总用时436.6秒，逐条用时不含模型加载。

16个WAV、manifest.jsonl、acoustic.json已从dub/out/test原样复制到本目录。清单覆盖、全部16条测量、音频可解码和非静音、复制前后SHA256均通过；结果见[verification.json](verification.json)、[生成清单](manifest.jsonl)、[声学测量](acoustic.json)、[运行输出](test-output.txt)。运行输出仅过滤加载/token进度条，原始日志保留在本机deployment/2026-10-22-dub-test.log。

## 环境与源码

已拉取`0ecdfee127c9a030c57c2dae797fa5972234c0ab`，阅读README第7节及voices/README.md。系统FFmpeg 8.1.1和ffprobe可用，按文档跳过imageio-ffmpeg；xvx-vox为Python 3.11.15、torch 2.11.0+cu128，CUDA 12.8可用、算力(12,0)。

参考音频均从已有报告找到；完整命令、实际输入、参考文件与源文件基线见[plan.json](plan.json)，当前阶段见[progress.json](progress.json)。specs.py、dub/xiaonian_v10.json、全部脚本及音色v1文件与拉取后SHA256一致；无Backend或脚本修改。同一时间仅运行一个模型。

## 视频和分轨位置

- 原版：`D:\XiaoNian-Video\review\2026-10-21-v10\晓念_V10_完整片_无旁白.mp4`；178秒、1920×1080、60fps，89,795,477字节。
- 后续预备使用的30MB以内版：`D:\XiaoNian-Video\review\2026-10-21-v10\晓念_V10_完整片_无旁白_30MB以内.mp4`；178.005秒、1920×1080、60fps，28,945,981字节。
- BGM：`D:\XiaoNian-Video\review\2026-10-21-v10\audio\music.wav`。
- 音效：`D:\XiaoNian-Video\review\2026-10-21-v10\audio\sfx.wav`。

两个分轨均为178秒、48kHz、24bit双声道，探测结果见[media.json](media.json)。正式脚本拷贝画面流并替换音轨，因此预备使用已压到30MB以内的版本；本轮未修改、复制或上传这些视频和底乐分轨。

## 盲听与停止点

用Edge打开工具包rater.html，点「打开文件夹（自动保存）」，选择dub/out/test。按「正式解说是否清楚、自然、有交流感」盲听，统计里的「方向」对应四个参考音色。等用户指定音色后才执行正式render，届时使用现成music.wav及sfx.wav；ASR首次下载失败时按用户要求加--no-asr重跑。

正式配音尚未开始，final报告、人声轨和配音版MP4均未生成；后续完成后才归档配音报告.md、report.json、人声轨_48k24bit.wav并转告「需要注意」原文，MP4只留在本机。
