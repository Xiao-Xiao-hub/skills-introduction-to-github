from pathlib import Path
import shutil
from c3_video_r3 import ROOT,OUT,FINAL,read,write,sha

report=read(OUT/'report.json');check=read(OUT/'encoded-asr.json')
assert check['match'],'Do not publish a passing claim for an unresolved text check'
assert sha(FINAL)==report['video_sha256']
for path,digest in read(ROOT/'reports/2026-10-22-dub-c3-expression/plan.json')['source_hashes'].items():assert sha(ROOT/path)==digest,path
dest=ROOT/'reports/2026-10-22-dub-v10-c3-r3';dest.mkdir(parents=True,exist_ok=True)
for name in ('report.json','selection.json','candidates.json','encoded-asr.json','人声轨_48k24bit.wav','替换段_2分21至2分33.wav'):
    shutil.copy2(OUT/name,dest/name)
for name in ('c3_video_r3.py','c3_video_r3_archive.py'):shutil.copy2(ROOT/'deployment'/name,dest/name)
selected=report['selected'];g=selected['neighbor_cosines']
text=f'''# 晓念 V10 C3 R3：修复技术段衔接，取消片尾旁白

用户反馈2:21–2:33音色和语调突变，要求迅速修订，并让片尾只展示画面。

- 重做23b三句完整语流，直接使用前一段23a的声音与文字作连续参考。上一版此处复用了旧C2较平直的素材，本轮不再使用该回退素材。
- 生成2个候选；选中seed {selected['seed']}，全段恒定速度{report['fit']['rate']:.3f}倍，无内部切片、无新增句间静音。与前后段的CAM++余弦为{g['23a']:.3f}、{g['24']:.3f}；仅用于身份漂移检查，不是人耳音色或自然度验收。
- 从169.5秒起人声轨严格为零，片尾释义和品牌仅展示画面，保留原BGM及音效。
- 其它旁白样本逐样本一致，保留R2音乐/音效混音策略，不新增背景压低。原视频流复制，未重新压缩画面。
- 选中干声与最终AAC的替换段分别通过Paraformer、SenseVoice文本核对；最终视频完整解码、178秒、1080p60、10680帧与视频包流哈希检查通过。
- 最终响度{report['loudness']['input_i']} LUFS，真峰值{report['loudness']['input_tp']} dBTP。

本机成片：`{FINAL}`，{report['video_bytes']:,}字节。MP4只留本机。

## 需要注意

本轮按用户指定只调整技术段并取消片尾旁白；音色衔接与自然度仍以实际试听为准。原片约72秒、154秒字幕与旁白的同义措辞差异保留。

未修改specs.py、dub/xiaonian_v10.json、common.py或原脚本。原素材、R1、R2均保留。完整人声、替换段与机器核对原文一并归档。
'''
(dest/'README.md').write_text(text,encoding='utf-8');(dest/'配音报告.md').write_text(text,encoding='utf-8')
with (ROOT/'PROGRESS.md').open('a',encoding='utf-8')as f:f.write('\n\n## 2026-10-22 C3 R3 技术段衔接与无旁白片尾\n\n按用户反馈重做2:21–2:33三句，以前一段声音作参考；取消169.5秒后的全部片尾旁白，保留原音乐音效与画面。其它人声样本一致，替换段实际AAC文本核对及成片技术检查通过，听感待用户确认。[报告与音轨](reports/2026-10-22-dub-v10-c3-r3/README.md)。\n')
print(dest)
