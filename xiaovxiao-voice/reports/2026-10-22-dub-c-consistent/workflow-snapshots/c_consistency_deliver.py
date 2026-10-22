"""Archive the accepted machine checks and standalone C2 narration."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import re
import shutil
import sys
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import dub
from common import read_manifest

OUT=ROOT/'dub/out/c-consistent-2026-10-22'
REPORT=ROOT/'reports/2026-10-22-dub-c-consistent'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

accepted=read(OUT/'acceptance.json')
if not accepted['machine_checks_passed']:raise SystemExit('Refusing delivery: machine checks failed')
source=OUT/'final/人声轨_48k24bit.wav'
if sha(source)!=accepted['waveform_sha256']:raise SystemExit('Verified audio changed')
wave=read(OUT/'final-waveform-verification.json');plan=read(REPORT/'plan.json')
selection=read(OUT/'selection.json');data=dub.load_script(dub.SCRIPT)
delivery=OUT/'delivery';delivery.mkdir(exist_ok=True)
filename='C2_沉稳一致_完整旁白.wav'
shutil.copy2(source,delivery/filename)
for name in ['人声轨_48k24bit.wav','配音报告.md','report.json']:
    shutil.copy2(OUT/'final'/name,REPORT/name)
for name in ['acceptance.json','final-waveform-verification.json','selection.json','candidate-review.json',
             'audit-groups-basic.json','audit-groups-speaker-summary.json',
             'basic-cache.meta.json','speaker-cache.meta.json','asr-cache.meta.json','cross-asr.json','cross-final-asr.json']:
    shutil.copy2(OUT/name,REPORT/name)
if len({sha(source),sha(delivery/filename),sha(REPORT/source.name)})!=1:raise SystemExit('Copied WAV differs')

all_generated={e['id']:e for folder in ['pilot','takes'] for e in read_manifest(OUT/folder)}
plan.update(stage='machine_passed_user_listening_pending',completed_at=datetime.now().astimezone().isoformat(timespec='seconds'),
            generated_success=len(all_generated),generation_failures=0,selected_groups=len(selection),
            seed_exceptions={g:v['seed'] for g,v in selection.items() if v['seed']!=41},
            delivery_file=str(delivery/filename),waveform_sha256=sha(source),
            original_scripts_changed=False,backend_changes=False,video_created=False)
write(REPORT/'plan.json',plan)
full_text='\n'.join(l['text'] for g in data['groups'] for l in g['lines'])
manifest={'id':'xiaonian_v10_C2_consistent','file':filename,'model':'voxcpm2','mode':'clone',
          'archetype':'C2 沉稳一致','variant':'完整旁白','text':full_text,
          'sr':48000,'duration':178.0,'gen':{'cfg':2.0,'steps':20},'created':plan['completed_at']}
(delivery/'manifest.jsonl').write_text(json.dumps(manifest,ensure_ascii=False)+'\n',encoding='utf-8')
report_manifest=dict(manifest,file='人声轨_48k24bit.wav')
(REPORT/'manifest.jsonl').write_text(json.dumps(report_manifest,ensure_ascii=False)+'\n',encoding='utf-8')

original_md=(OUT/'final/配音报告.md').read_text(encoding='utf-8')
warnings=re.search(r'## 需要注意\s*\n(.*?)(?=\n## |\Z)',original_md,re.S)[1].strip()
(REPORT/'需要注意_原文.md').write_text('# 原脚本的「需要注意」原文\n\n'+warnings+'\n\n这些为整组汉字字面CER提示。最终55句另行提取识别，原始文本和同音字异写均保留在final-waveform-verification.json；未把原脚本警告删除。关键多音字的实际读音和真人感仍需要听觉确认。\n',encoding='utf-8')
o=accepted['old_C_final'];n=accepted['new_C_final']
normalized=[{'cue':r['index'],'group':r['group'],'expected':r['text'],'recognized':r['asr']['text'],'raw_cer':r['raw_cer']} for r in wave['cues'] if r['raw_cer']>0]
write(REPORT/'同音字识别差异.json',normalized)
write(REPORT/'delivery.json',{'file':str(delivery/filename),'bytes':source.stat().st_size,'sha256':sha(source),
    'duration':178,'sample_rate':48000,'subtype':'PCM_24','channels':1,'machine_checks_passed':True,
    'human_listening_passed':False,'report_audio':'人声轨_48k24bit.wav'})
evidence=REPORT/'workflow-snapshots';evidence.mkdir(exist_ok=True)
for name in ['c_consistency_generate.py','c_consistency_check.py','c_consistency_select.py','c_consistency_final_check.py','c_consistency_gate.py','c_consistency_cross_asr.py','c_consistency_cross_final.py','c_consistency_deliver.py']:
    shutil.copy2(ROOT/'deployment'/name,evidence/name)
(evidence/'README.txt').write_text('本目录为本轮外部编排/验收辅助程序快照，未修改工具包原脚本。程序ROOT按deployment目录定位。复现时应放回工具包deployment/，从工具包根用xvx-vox运行；运行数据路径见plan.json。已通过的成品以delivery.json的SHA256为准，不要重新自动选条覆盖后冒称同一成品。\n',encoding='utf-8')
summary=f'''# 晓念 V10：C2 沉稳一致版

完成时间：{plan['completed_at']}。完整纯旁白已交付并通过本轮机器验收；听感验收等待用户完整试听。原C保留。

## 音频

[完整旁白：178秒 / 48kHz / 24bit / 单声道](人声轨_48k24bit.wav)，{source.stat().st_size/1e6:.2f} MB。从视频00:00对齐，不含BGM、音效或视频。

本机：`{delivery/filename}`。文件SHA256：`{sha(source)}`。

## 解决的问题

旧C每组用了不同语气指令，自动选条未检查跨段音色。此次仍用VoxCPM2，但改用官方高保真克隆：冻结旧C研究说明段为单一prompt/reference，提供准确文字，全片统一cfg2/20步，初始seed41。仅{len(plan['seed_exceptions'])}个未通过候选或切后文字门槛的组换用补生成结果，具体seed见selection.json。无逐段情绪重设，无音高变调处理，原文55句不变。

先比较3个参考、9条同文小样；所选参考3/3字面ASR正确，语速适中，声纹稳定。其他参考出现语速偏快或识别差异。共新生成{len(all_generated)}条、生成失败0条；选择24组构成55句完整旁白。

## 最终音轨复核

所有数字比较均取旧C与新C最终音轨的同样24组，使用同一分析器。

| 指标 | 旧C | 新C2 |
|---|---:|---:|
| 各组平均音高范围 | {o['mean_f0_min']:.0f}–{o['mean_f0_max']:.0f} Hz | {n['mean_f0_min']:.0f}–{n['mean_f0_max']:.0f} Hz |
| 组间平均音高标准差 | {o['mean_f0_sd']:.1f} Hz | {n['mean_f0_sd']:.1f} Hz |
| 相邻组最大音高变化 | {o['mean_f0_adjacent_max_semitones']:.2f} 半音 | {n['mean_f0_adjacent_max_semitones']:.2f} 半音 |
| CAM++段间余弦均值 | {o['speaker_pairwise_mean']:.3f} | {n['speaker_pairwise_mean']:.3f} |
| CAM++段间余弦最低值 | {o['speaker_pairwise_min']:.3f} | {n['speaker_pairwise_min']:.3f} |
| CAM++段间余弦P10 | {o['speaker_pairwise_p10']:.3f} | {n['speaker_pairwise_p10']:.3f} |
| 语速范围 | {o['rate_min']:.1f}–{o['rate_max']:.1f} 字/秒 | {n['rate_min']:.1f}–{n['rate_max']:.1f} 字/秒 |

- 55/55句：从最终音轨提取后逐句ASR核对；除明确列出的同音字异写外无文字差异，未发现漏词、重复或串句。
- 另用SenseVoiceSmall对最终14个包含关键词或识别分歧的短句独立复核，均通过同音异写归一化比较。没有向识别模型提供稿子或热词。“报竞赛”和“长任务”的候选因识别分歧更换，交叉结果保留在cross-asr.json及cross-final-asr.json。
- 24/24组通过预先记录的参考相似度、段间相似度、F0偏移/分布、HNR、语速门槛；保留句内自然语调变化，没有把音高拉成直线。
- 时间：55句均可靠切开，无整组回退；无超限起点顺延、侵入下一句或末尾截断。最大时间适配{max(l['speed'] for l in read(OUT/'final/report.json')['lines']):.3f}倍。
- 格式：178.000秒、48kHz、PCM24单声道；峰值{wave['peak_dbfs']:.2f}dBFS，未达到满幅。
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
'''
(REPORT/'README.md').write_text(summary,encoding='utf-8')
(delivery/'试听说明.txt').write_text('C2 沉稳一致版：178秒完整纯旁白，48kHz/24bit单声道，从视频00:00对齐。\n逐句与音色一致性机器验收已通过，最终自然听感请完整试听。\n原C保留，本目录没有BGM或视频。\n',encoding='utf-8')
progress=ROOT/'PROGRESS.md'
with progress.open('a',encoding='utf-8') as f:
    f.write(f"\n\n## 2026-10-22 C2沉稳一致版（机器验收通过）\n\n已按用户反馈完成C版重制：固定C研究说明段高保真克隆，24组/55句/178秒。共新生成{len(all_generated)}条，合成失败0条。最终同条件比较CAM++段间均值{o['speaker_pairwise_mean']:.3f}→{n['speaker_pairwise_mean']:.3f}，最低{o['speaker_pairwise_min']:.3f}→{n['speaker_pairwise_min']:.3f}；组均音高范围{o['mean_f0_min']:.0f}–{o['mean_f0_max']:.0f}→{n['mean_f0_min']:.0f}–{n['mean_f0_max']:.0f}Hz。55句切后ASR与时间门槛通过，同音异写详列，人工听感仍待用户确认。原稿/脚本/Backend未改，未合视频。详见[报告与音频](reports/2026-10-22-dub-c-consistent/README.md)。\n")
print(json.dumps({'delivery':str(delivery/filename),'bytes':source.stat().st_size,'sha256':sha(source),'generated':len(all_generated),'checks':accepted['checks']},ensure_ascii=False,indent=2))
