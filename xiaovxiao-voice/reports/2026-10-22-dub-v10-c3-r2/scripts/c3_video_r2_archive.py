"""Archive the R2 review cut and its measured changes; MP4 remains local."""
from pathlib import Path
import shutil
import numpy as np
from c3_video_align import ROOT, read, write, sha
from c3_video_r2_plan import OUT

reportdir=ROOT/'reports/2026-10-22-dub-v10-c3-r2'
mix=read(OUT/'mix-report.json');asr=read(OUT/'clean-group-asr.json');encoded=read(OUT/'final-aac-check.json');alignment=read(OUT/'aligned/alignment.json')
assert asr['cross_checked_passed']
encoded_matches=sum(any(row.get(engine,{}).get('match',False) for engine in ('sensevoice','paraformer')) for row in encoded['rows'])
encoded_unresolved=[row['group'] for row in encoded['rows'] if not any(row.get(engine,{}).get('match',False) for engine in ('sensevoice','paraformer'))]
assert asr['waveform_sha256']==alignment['waveform_sha256']==mix['voice_sha256']
assert encoded['video_sha256']==mix['video_sha256']==sha(mix['video'])
for path,digest in read(ROOT/'reports/2026-10-22-dub-c3-expression/plan.json')['source_hashes'].items():assert sha(ROOT/path)==digest,path
old=read(OUT/'pause-analysis.json')
body=asr['cue_timing'][:-3]
summary=dict(status='complete_review_cut_user_listening_pending',video=mix['video'],video_bytes=mix['video_bytes'],video_sha256=mix['video_sha256'],
    voice_sha256=mix['voice_sha256'],duration=178,resolution=[1920,1080],fps=60,
    old_mean_group_internal_pause=old['overall']['final_reasr_gap_seconds']['mean'],new_mean_group_internal_pause=asr['mean_body_internal_gap_seconds'],
    comparable_boundaries=len(asr['body_internal_gaps']),internal_cuts=0,inserted_internal_silence=0,
    primary_group_matches=asr['primary_matches'],clean_cross_checked_groups=24,encoded_checked_groups=len(encoded['rows']),encoded_matched_groups=encoded_matches,
    encoded_cross_checked_passed=encoded['cross_checked_passed'],encoded_unresolved_groups=encoded_unresolved,
    max_body_asr_early_seconds=max(r['early_seconds']for r in body),max_body_asr_late_seconds=max(r['late_seconds']for r in body),
    actual_closing_asr=[r['actual_asr_speech']for r in asr['cue_timing'][-3:]],background_comparison=encoded['background_comparison'],
    voice_to_background_target_db=mix['voice_over_bed_db'],music_duck_db=0,sfx_duck_db=0,closing_extra_duck_db=0,
    final_lufs=float(mix['final_aac_loudness']['input_i']),true_peak_dbtp=float(mix['final_aac_loudness']['input_tp']),
    picture_stream_unchanged=mix['validation']['checks']['video_stream_hash_matches'],original_scripts_unchanged=True,human_listening=False)
write(reportdir/'report.json',summary)
files={OUT/'aligned/alignment.json':'alignment.json',OUT/'clean-group-asr.json':'clean-group-asr.json',OUT/'final-aac-check.json':'final-aac-check.json',
       OUT/'mix-report.json':'mix-report.json',OUT/'body-group-plan.json':'body-group-plan.json',OUT/'pause-analysis.json':'pause-analysis.json',
       OUT/'closing/r2-closing-results.json':'closing-candidates-first-four.json',OUT/'aligned/人声轨_48k24bit.wav':'人声轨_48k24bit.wav'}
closing=alignment['groups'][-1]
write(reportdir/'selected-closing.json',closing)
files[Path(closing['source']['path'])]='片尾_完整重录原素材.wav'
files[Path(closing['rendered_path'])]='片尾_完整连续语流.wav'
for p in (OUT/'closing').glob('*plain*.json'):files[p]=p.name
for source,name in files.items():
    shutil.copy2(source,reportdir/name);assert sha(source)==sha(reportdir/name)
scripts=reportdir/'scripts';scripts.mkdir(exist_ok=True)
for source in (ROOT/'deployment').glob('c3_video_r2_*.py'):shutil.copy2(source,scripts/source.name)
history=reportdir/'history';history.mkdir(exist_ok=True)
for folder in OUT.iterdir():
    if folder.is_dir() and folder.name.startswith('before-'):
        target=history/folder.name;target.mkdir(exist_ok=True)
        for source in folder.glob('*.json'):shutil.copy2(source,target/source.name)
checks=reportdir/'checks';checks.mkdir(exist_ok=True)
for p in (OUT/'checks').glob('*.jpg'):shutil.copy2(p,checks/p.name)
write(checks/'visual-timing.json',dict(source_video='D:/XiaoNian-Video/review/2026-10-21-v10/晓念_V10_完整片_无旁白.mp4',
    source_code=['D:/XiaoNian-Video/work-v10/film/src/v10/timeline.ts:24','D:/XiaoNian-Video/work-v10/film/src/v5/scenes.tsx:781','D:/XiaoNian-Video/work-v10/film/scripts/finish-v10.mjs:7'],
    definition_text_full_visible_until=174.5,definition_text_fully_transparent_at=174.859,brand_first_nonzero_frame_about=175.05,brand_full_visible_at=175.308,
    assistant_viewed_frames=[174.4,175.1],observations=['174.4: second definition is fully readable.','175.1: large 晓念 brand is visibly readable.'],human_playback=False))
bg=summary['background_comparison']
md=f'''# 晓念 V10 · C3 R2 音乐增强与连续语流修订

根据用户明确否定上一版听感的反馈完成修订。交付新的完整成片，旧版本保留。本轮机器检查针对漏字、时间与文件完整性，不能代替用户对情感和真人感的判断。

## 交付

- 本机成片：`{mix['video']}`
- 178 秒、1920×1080、60 fps、{mix['video_bytes']:,} 字节（{mix['video_bytes']/1e6:.2f} MB）。原片视频包流哈希一致，画面未重新压缩；MP4 不上传 GitHub。
- [完整独立人声](人声轨_48k24bit.wav)、[片尾连续语流](片尾_完整连续语流.wav)、[完整证据](report.json)。

## 针对反馈的改动

1. 背景：music/sfx保留原始包络、位置与音高，取消逐句 ducking 和片尾额外降音量；人声对背景目标从上版10dB改为{mix['voice_over_bed_db']}dB。实际最终PCM中，消除居中人声后的立体声背景侧向分量，正文提高{bg[0]['increase_db']:.2f}dB、片尾提高{bg[1]['increase_db']:.2f}dB。此为背景一个分量的实测，非主观响度评分。
2. 停顿：恢复完整C3组波形，正文23组内部零剪切、零新增静音，以整组恒定保音高调速对齐字幕。此前把连续音切开再插0.678秒静音的问题已消除。29个相同句界的ASR间隔均值由{summary['old_mean_group_internal_pause']:.3f}秒降至{summary['new_mean_group_internal_pause']:.3f}秒；完整波形与原有轻重起伏保留。
3. 片尾：同一C3 warm表达参考，完整生成三句，保持原速和自然连贯的介绍语气；没有独立补录并粘贴“晓念”。整段按照实际画内字与品牌动画安排，品牌声从标题开始可见时进入。一次整体慢放导致读音识别恶化，已撤销并保留失败证据。正文一组23b也保留原始波形与速度，避免无意义的0.3%处理。完整片尾人声轻推2.5dB，背景保持。

## 检查结果

- 清洁人声24/24组交叉文本核对，主识别器{asr['primary_matches']}/24，其余由独立识别器复核；所有原文和分歧保留，不增添“想→晓”“明亮→明了”等强行归一规则。
- 最终AAC中选7个完整段落（包括旧硬切位置、长停顿位置、片尾等）复核，{encoded_matches}/7按既有同音字归一规则匹配。片尾干声两种识别器均核对通过，混音后两种识别器都把末尾“晓念”识别为“想念”，其中一种还把释义句的“晓”识别为“想”；此项未通过，不能据此宣称成片文本全面验收通过，保留原始结果供试听复核。
- 正文按实际字幕窗口附近保持自然语流，不再要求每句都从字幕起点开声；ASR估计最早超前{summary['max_body_asr_early_seconds']*1000:.0f}ms、最晚延后{summary['max_body_asr_late_seconds']*1000:.0f}ms。ASR词边界存在估计误差。
- 片尾按实际画面安排：释义文字完整显示到174.50秒，品牌约175.05秒开始可见；抽看174.4与175.1秒帧。保留同一次录音的自然间隔，不为旧旁白起点人为插长停顿。
- 最终AAC {summary['final_lufs']:.2f} LUFS，真峰值 {summary['true_peak_dbtp']:.2f} dBTP；完整解码、1080p60/10680帧/178秒、视频包流哈希检查通过。

## 需要注意

1. 片尾干声核对通过，但混音后的“晓念”仍被两种识别器识别成“想念”；请重点听最后几秒的品牌读音。
2. 本次恢复了背景音乐并减少了硬切停顿；情感、真人感与最终音量平衡仍需你看片判断。
3. 原片约72秒、154秒的两处字幕与定稿旁白同义措辞差异继续保留，未改画面和台词。

## 来源与保留

原片、原music/sfx、批准C3、首版成片均保留。未修改specs.py、dub/xiaonian_v10.json、common.py及工具包原脚本；只新增独立编排与报告。模型始终串行运行，未安装新的包或模型。

采用本机现有FFmpeg/Rubber Band，官方方法参考（2026-10-22）：https://breakfastquay.com/rubberband/ 。具体命令与本次脚本快照见scripts/。模型/ASR分歧以及中间失败记录保留，未把它们改写为全通过。
'''
(reportdir/'README.md').write_text(md,encoding='utf-8');(reportdir/'配音报告.md').write_text(md,encoding='utf-8')
with (ROOT/'PROGRESS.md').open('a',encoding='utf-8') as f:
    f.write(f'\n\n## 2026-10-22 R2 音乐与连续语流成片待试听\n\n按用户反馈恢复音乐/音效原有包络并取消ducking，恢复整段语流，29处句间间隔均值0.923→{summary["new_mean_group_internal_pause"]:.3f}秒；片尾一次完整生成，不粘贴独立品牌词。清洁人声24组交叉核对通过；最终AAC抽查{encoded_matches}/7组匹配，混音片尾的“晓念”被两种识别器识别成“想念”，保留原文并标记待试听，未宣称全面验收。178秒1080p60原视频流复制，完整解码通过。旧版本保留，情感与真人感待用户看片判断。[R2交付与证据](reports/2026-10-22-dub-v10-c3-r2/README.md)。\n')
print(summary)
