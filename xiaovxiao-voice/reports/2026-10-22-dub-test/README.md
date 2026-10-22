# 晓念 V10 旁白试配

更新时间：2026-10-22T10:24:47+08:00（北京时间）。状态：文档、环境、媒体和参考音色均核对通过，正在运行16条试配；正式配音尚未开始。

已拉取`0ecdfee127c9a030c57c2dae797fa5972234c0ab`，阅读README第7节及voices/README.md。系统FFmpeg 8.1.1和ffprobe可用，按文档跳过imageio-ffmpeg；xvx-vox仍为torch 2.11.0+cu128、CUDA12.8可用、算力(12,0)。四个参考音色均找到，实际生成统一使用VoxCPM2，MOSS来源参考只作为克隆音频。

计划为4个参考音色 × 4组（05-06、07、14、23a） × style方式 × seed 1，共16条，CFG 2.0、10步。完整命令、参考文件、输入及源文件SHA256基线见[plan.json](plan.json)，阶段进度见[progress.json](progress.json)。同一时间只运行一个模型。

## 视频和分轨

- 178秒原版：`D:\XiaoNian-Video\review\2026-10-21-v10\晓念_V10_完整片_无旁白.mp4`，1920×1080、60fps、89,795,477字节。
- 30MB以内版：`D:\XiaoNian-Video\review\2026-10-21-v10\晓念_V10_完整片_无旁白_30MB以内.mp4`，约178秒、1920×1080、60fps、28,945,981字节。
- BGM：`D:\XiaoNian-Video\review\2026-10-21-v10\audio\music.wav`；音效：`D:\XiaoNian-Video\review\2026-10-21-v10\audio\sfx.wav`，均178秒、48kHz、24bit双声道。

媒体探测数据见[media.json](media.json)。后续正式配音预备使用30MB以内现成视频及两个分轨；当前不运行render，不复制或上传视频及底乐分轨。

## 盲听停止点

试配完成并归档后，用Edge打开工具包rater.html，点「打开文件夹（自动保存）」，选择dub/out/test。按「正式解说是否清楚、自然、有交流感」盲听，统计里的「方向」对应四个参考音色。等待用户指定音色后再正式配音，届时再按要求归档final报告、人声轨并转告「需要注意」。
