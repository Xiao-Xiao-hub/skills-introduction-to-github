"""Archive the finished local-only video evidence and voice, never the MP4."""
from datetime import datetime
from pathlib import Path
import shutil
from c3_video_align import OUT, ROOT, read, write, sha

reportdir=ROOT/'reports/2026-10-22-dub-v10-c3'
alignment=read(OUT/'aligned/alignment.json')
primary=read(OUT/'aligned/cue-verification.json')
secondary=read(OUT/'aligned/secondary-cue-verification.json')
final_asr=read(OUT/'final-aac-speech-check.json')
mix=read(OUT/'mix-report.json')
video=Path(mix['outputs']['final_mp4'])
voice=OUT/'aligned/人声轨_48k24bit.wav'
assert sha(voice)==alignment['waveform_sha256']==primary['waveform_sha256']==secondary['waveform_sha256']
assert sha(video)==final_asr['video_sha256']
assert mix['status']=='passed' and final_asr['passed']
for name,digest in alignment['original_source_hashes'].items():
    assert sha(ROOT/name)==digest,name
secondary_by_id={r['index']:r for r in secondary['rows']}
resolutions=[]
for row in primary['rows']:
    assert all(v for k,v in row['checks'].items() if k!='text')
    if not row['checks']['text']:
        s=secondary_by_id[row['index']]
        assert s['match']
        resolutions.append(dict(index=row['index'],expected=row['synth'],primary=row['asr']['text'],secondary=s['plain'],
                                decision='Secondary matches expected on the actual aligned waveform. Primary disagreement remains recorded; no new normalization rule added.'))
closing=read(OUT/'closing/selected.json')
closing.update(status='selected_for_delivery_with_disclosed_asr_disagreement',passed=True,
               acceptance_scope='Actual aligned and encoded final cues cross-checked. This is not unanimous ASR or human emotion/naturalness approval.',
               actual_aligned_wave_sha256=sha(voice),primary_disagreements=resolutions)
if alignment.get('brand_replacement'):
    closing['final_brand_replacement']=alignment['brand_replacement']
write(OUT/'closing/selected.json',closing)
deliverables=dict(video=str(video),video_bytes=video.stat().st_size,video_sha256=sha(video),
                  voice=str(voice),voice_sha256=sha(voice),voice_bytes=voice.stat().st_size)
summary=dict(status='delivered_technical_checks_complete_user_listening_pending',date=datetime.now().isoformat(),
             deliverables=deliverables,cues=55,inside_windows=55,primary_asr_matches=55-len(resolutions),
             secondary_resolutions=resolutions,final_aac_selected_cues=final_asr['indices'],
             final_aac_selected_cues_match=True,final_aac_lufs=float(mix['loudness']['final_aac_measured']['input_i']),
             final_aac_true_peak_dbtp=float(mix['loudness']['final_aac_measured']['input_tp']),
             picture_stream_unchanged=mix['validation']['checks']['video_stream_hash_matches'],
             source_files_unchanged=True,original_code_unchanged=True,human_listening=False,
             emotion_naturalness='New warm reference and expressive generation direction; not established by pitch, ASR emotion tags, or speaker embeddings.',
             subtitle_wording_differences=alignment['subtitle_wording_differences'])
write(reportdir/'report.json',summary)
files={
    OUT/'aligned/alignment.json':'alignment.json',
    OUT/'aligned/cue-verification.json':'cue-verification.json',
    OUT/'aligned/secondary-cue-verification.json':'secondary-cue-verification.json',
    OUT/'final-aac-speech-check.json':'final-aac-speech-check.json',
    OUT/'mix-report.json':'mix-report.json',
    OUT/'checks/visual-check.json':'visual-check.json',
    OUT/'closing/candidates.json':'closing-candidates.json',
    OUT/'closing/selected.json':'selected-closing.json',
    OUT/'closing/no-whole-group-candidate-passed.json':'initial-whole-group-selection-failure.json',
    OUT/'closing/interrupted-old-c2-reference-batch.json':'initial-reference-error.json',
    Path(closing['path']):'片尾三句_新生成原素材.wav',
    voice:'人声轨_48k24bit.wav',
}
if alignment.get('brand_replacement'):
    brand=alignment['brand_replacement']['candidate_evidence']
    write(reportdir/'selected-brand.json',brand)
    files[Path(brand['path'])]='片尾品牌名_重录原素材.wav'
    files[OUT/'closing/brand-tag/brand-tag-results.json']='brand-tag-results.json'
for source,name in files.items():
    shutil.copy2(source,reportdir/name)
    assert sha(source)==sha(reportdir/name)
for folder in ['first-mix-before-clarity-adjustment','second-mix-before-brand-replacement']:
    source=OUT/folder
    if source.exists():
        target=reportdir/'history'/folder;target.mkdir(parents=True,exist_ok=True)
        for p in source.glob('*.json'):shutil.copy2(p,target/p.name)
scripts=reportdir/'scripts';scripts.mkdir(exist_ok=True)
for p in (ROOT/'deployment').glob('c3_video_*.py'):
    shutil.copy2(p,scripts/p.name)
plan=read(reportdir/'plan.json');plan.update(stage=summary['status'],deliverables=deliverables,
    acceptance_note='The primary recognizer is not unanimous. Its actual disagreements are retained and independently cross-checked; no 想→晓 or 明亮→明了 normalization was introduced.')
write(reportdir/'plan.json',plan)
md=f'''# 晓念 V10 · C3 片尾情感调整成片

已导出完整成片。主体沿用用户选定 C3，片尾三句重新生成；最后的短品牌名进一步独立重录。使用同一 C3 warm 表达参考，情绪方向为真诚温暖、递进、最后温柔笃定地收住。自然度与情感是否合意仍由用户试听判断。

## 交付

- 本机视频：`{video}`
- 视频大小：{video.stat().st_size:,} 字节（{video.stat().st_size/1e6:.2f} MB）；178 秒、1920×1080、60 fps、10680 帧。
- 原画面码流复制，与用户指定原片的视频包流 SHA256 一致；没有重新压缩画面。MP4 不推 GitHub。
- [完整独立人声](人声轨_48k24bit.wav)：178 秒、48 kHz、单声道 PCM24。
- [完整机器证据](report.json)、[逐句对齐](alignment.json)、[混音报告](mix-report.json)。

## 实际检查结果

- 55/55 句起点按源字幕毫秒时间或画内字时间放置，全部在对应窗口内；主体使用已定 C3 波形，没有重生。最大变速 {alignment['max_speed']:.5f} 倍；仅在必要位置轻微调速，不改音高。
- 剪后人声主识别器 {55-len(resolutions)}/55 句归一化匹配；另 {len(resolutions)} 句由 SenseVoice 在实际剪后音频上复核匹配。主识别器的原始分歧留存，未伪写为两套识别器全数一致。
- 最终 MP4 中选择片尾三句、分歧正文句，以及人声对背景比最低的三句，共 {len(final_asr['indices'])} 句实际解码复核，{len(final_asr['indices'])}/{len(final_asr['indices'])} 匹配。
- 保留原 music.wav 与 sfx.wav 的内容、位置、音高。旁白时音乐平滑降低 8 dB、音效 2.5 dB；片尾进一步降低背景 6 dB，随后恢复。
- 最终 AAC 实测 {summary['final_aac_lufs']:.2f} LUFS，真峰值 {summary['final_aac_true_peak_dbtp']:.2f} dBTP；完整解码通过，原片及分轨哈希不变。
- 抽看原片 13 帧，包括片尾三处画内字；输出视频流相同。未声称人工完整观看或听感验收通过。

## 需要注意

1. 原片约 72 秒、154 秒各有一处字幕与旁白的同义措辞差异，时间已对齐，画面保留原样。
2. 两处语音识别分歧已交叉复核；情感和自然度仍需你看片判断。

## 差异与制作记录

71.845–74.673 秒字幕为“相关研究中，直接给答案组”，旁白为“相关研究中，直接给答案的那组”；153.180–156.598 秒字幕为“导入离开后，回来可以继续处理”，旁白为“导入时离开，回来还能继续处理”。

第 10、54 句的主识别器分歧见下表。初混片尾品牌名在背景中出现识别歧义，因此保留失败证据、进一步降低片尾背景并独立重录最后两个字。旧成片和音轨均留在本机历史目录。

早期生成助手误用旧 C2 参考，已中止并重做；其中三条临时输出在记录哈希前因文件名碰撞被新批次覆盖，不能冒称保留了旧波形证据，见 initial-reference-error.json。交付使用的 C3 参考及音频均有实际哈希。

| 句号 | 定稿 | 主识别器原文 | 第二识别器原文 |
|---|---|---|---|
'''
for r in resolutions:md+=f"| {r['index']} | {r['expected']} | {r['primary']} | {r['secondary']} |\n"
md+='''
## 复现与边界

原 specs.py、dub/xiaonian_v10.json、common.py 与工具包原脚本均未修改。独立编排脚本留在 scripts/，模型始终串行运行；原 C3、原片、原 music/sfx 均保留。10 条片尾有效参考候选的原始结果留在 closing-candidates.json，失败候选未冒充通过。

复用已有 FFmpeg/FunASR：
- https://ffmpeg.org/ffmpeg-filters.html#loudnorm
- https://ffmpeg.org/ffmpeg-filters.html#atempo
- https://github.com/modelscope/FunASR/blob/main/docs/python_api.md

上面来源用于实现方法参考，实际验收以本目录音频、哈希和机器结果为准。采集日期 2026-10-22。
'''
(reportdir/'README.md').write_text(md,encoding='utf-8')
(reportdir/'配音报告.md').write_text(md,encoding='utf-8')
progress=ROOT/'PROGRESS.md'
with progress.open('a',encoding='utf-8') as f:
    f.write(f'\n\n## 2026-10-22 C3 片尾情感调整与 V10 成片完成\n\n主体保留用户选定 C3；片尾三句新生成、最后品牌名独立重录。55 句按原字幕/画内字时间对齐，原片视频流 SHA256 一致；178 秒1080p60，原 music/sfx 保留并平滑混音。成片 AAC 实测 {summary["final_aac_lufs"]:.2f} LUFS / {summary["final_aac_true_peak_dbtp"]:.2f} dBTP、完整解码通过。剪后主 ASR {55-len(resolutions)}/55，其余 {len(resolutions)} 句独立交叉核对；最终 AAC 抽查 {len(final_asr["indices"])}/{len(final_asr["indices"])} 匹配。识别分歧和原片两处字幕同义差异均披露，人工情感听感待用户看片。视频只在本机，完整人声和证据同步至 [本轮报告](reports/2026-10-22-dub-v10-c3/README.md)。\n')
print(summary)
