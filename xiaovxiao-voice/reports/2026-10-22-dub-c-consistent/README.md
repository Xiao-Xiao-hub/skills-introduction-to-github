# 晓念 V10：C2 沉稳一致版

完成时间：2026-10-22T14:30:30+08:00。完整纯旁白已交付并通过本轮机器验收；听感验收等待用户完整试听。原C保留。

## 音频

[完整旁白：178秒 / 48kHz / 24bit / 单声道](人声轨_48k24bit.wav)，25.63 MB。从视频00:00对齐，不含BGM、音效或视频。

本机：`E:\skills-introduction-to-github\xiaovxiao-voice\dub\out\c-consistent-2026-10-22\delivery\C2_沉稳一致_完整旁白.wav`。文件SHA256：`413d11f21c8137e99aa95586daf317a77d75a25296372a7f43f83995989d8186`。

## 解决的问题

旧C每组用了不同语气指令，自动选条未检查跨段音色。此次仍用VoxCPM2，但改用官方高保真克隆：冻结旧C研究说明段为单一prompt/reference，提供准确文字，全片统一cfg2/20步，初始seed41。仅4个未通过候选或切后文字门槛的组换用补生成结果，具体seed见selection.json。无逐段情绪重设，无音高变调处理，原文55句不变。

先比较3个参考、9条同文小样；所选参考3/3字面ASR正确，语速适中，声纹稳定。其他参考出现语速偏快或识别差异。共新生成38条、生成失败0条；选择24组构成55句完整旁白。

## 最终音轨复核

所有数字比较均取旧C与新C最终音轨的同样24组，使用同一分析器。

| 指标 | 旧C | 新C2 |
|---|---:|---:|
| 各组平均音高范围 | 196–318 Hz | 194–233 Hz |
| 组间平均音高标准差 | 32.4 Hz | 9.6 Hz |
| 相邻组最大音高变化 | 6.70 半音 | 2.30 半音 |
| CAM++段间余弦均值 | 0.760 | 0.925 |
| CAM++段间余弦最低值 | 0.548 | 0.850 |
| CAM++段间余弦P10 | 0.656 | 0.896 |
| 语速范围 | 4.0–7.0 字/秒 | 4.7–6.0 字/秒 |

- 55/55句：从最终音轨提取后逐句ASR核对；除明确列出的同音字异写外无文字差异，未发现漏词、重复或串句。
- 另用SenseVoiceSmall对最终14个包含关键词或识别分歧的短句独立复核，均通过同音异写归一化比较。没有向识别模型提供稿子或热词。“报竞赛”和“长任务”的候选因识别分歧更换，交叉结果保留在cross-asr.json及cross-final-asr.json。
- 24/24组通过预先记录的参考相似度、段间相似度、F0偏移/分布、HNR、语速门槛；保留句内自然语调变化，没有把音高拉成直线。
- 时间：55句均可靠切开，无整组回退；无超限起点顺延、侵入下一句或末尾截断。最大时间适配1.023倍。
- 格式：178.000秒、48kHz、PCM24单声道；峰值-1.01dBFS，未达到满幅。
- 来源：specs.py、dub/xiaonian_v10.json、所有原脚本和v1声音文件哈希未变；Backend未改；torch2.11.0+cu128保留。未安装flash-attn，模型顺序运行。

证据：[验收总表](acceptance.json)、[55句识别/时间检查](final-waveform-verification.json)、[选条溯源](selection.json)、[全部候选准入](candidate-review.json)、[同音字差异](同音字识别差异.json)、[配音报告](配音报告.md)、[原脚本警告原文](需要注意_原文.md)。

## 验收范围

机器检查通过不等于人耳确认了真人感。此次未冒称人工试听通过；“晓念、小狸、校对、明了、AI、SQLite”的实际读音、呼吸和正式介绍的听感，仍需用户完整试听确认。CAM++分数是余弦相似度，不是相似百分比或身份认证结论。本轮工程门槛记录在plan.json，不是官方自然度评级。

首条小样曾因单次辅助程序少写gen_label导致日志打印退出，已保存的音频保留、补齐元数据后续跑，未发生模型合成失败，也未修改原脚本。第一版未通过的完整拼接保留在本机rejected-first-assembly，不作为交付。

## 官方复用来源（2026-10-22）

- [VoxCPM](https://github.com/OpenBMB/VoxCPM)：prompt音频/准确文字/reference组合克隆。
- [FunASR](https://github.com/modelscope/FunASR/blob/main/docs/speaker_emotion.md)、[CAM++模型](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common)：说话人嵌入，CPU检测。
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice)：另一套独立中文识别模型，复核关键短句。
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)：调研Base作为后备；VoxCPM2小样与全片筛选通过，未下载或运行Qwen Base。

复现材料：[编排与验收程序快照](workflow-snapshots/)、plan.json、固定参考音频/文字。原始takes与缓存保留本机，不重复上传全部候选。
