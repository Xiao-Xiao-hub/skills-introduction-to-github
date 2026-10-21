# 音色设定 v2 · 参数校准 r0

更新时间：2026-10-22T00:13:33+08:00（北京时间）。状态：运行中，尚未完成验收。

## 范围与进度

已在分支 `claude/nifty-ritchie-n04rar` 拉取到 `a37e7072c74b962028b34ee2f603c3913b8b70f8`，阅读 README 第 1、2、3 节。两环境测量库及 CUDA 检查均通过，详见 [environment.json](environment.json)。

- VoxCPM2：A/B/D/F × 中文描述 × seed 1 × CFG 1.5/2.0 × 步数 10/20，共 16 条，正在运行。
- Qwen3：A/B/D/F × 中文描述 × seed 1 × temperature 0.9/1.0，共 8 条，等待 VoxCPM2 进程退出后运行。
- 输出为工具包内 `rounds/r0`。生成结束后再显式运行 `.\.venvs\xvx-vox\Scripts\python.exe measure.py rounds\r0`，保存并原样报告控制台输出。
- 最终将全部 WAV、manifest.jsonl、acoustic.json 原样复制到本目录，保持 manifest 的音频相对路径可用。

参数与源文件基线见 [plan.json](plan.json)。本轮不修改 specs.py 或脚本逻辑，不并行加载两个模型。

## 人工验收与后续

当前未评分、未选定海选参数，正式海选未启动。校准完成后，用 Edge 打开 rater.html，点「打开文件夹（自动保存）」，选择工具包内 rounds/r0，开始盲听打分。

用户说「评完了」之后，再复制 rounds/r0/ratings.json 到本目录并提交、推送。收到具体参数之前不启动 120 条海选。声学数字只辅助理解样本，主观听感以用户评分为准。
