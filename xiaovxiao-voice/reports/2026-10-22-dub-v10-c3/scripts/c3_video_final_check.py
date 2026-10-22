"""Check final encoded MP4 speech where masking matters most, plus revised closing."""
import json
import re
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from c3_video_align import OUT, ROOT, read, write, sha, norm, dub

report=read(OUT/'mix-report.json')
alignment=read(OUT/'aligned/alignment.json')
video=Path(report['outputs']['final_mp4'])
weakest=sorted(report['per_cue'],key=lambda x:x['voice_rms_dbfs']-x['ducked_bed_rms_dbfs'])[:3]
indices=sorted({10,53,54,55}|{x['index'] for x in weakest})
with tempfile.TemporaryDirectory(prefix='c3-final-aac-check-') as temp:
    wave=Path(temp)/'decoded.wav'
    dub.run_ffmpeg('-i',video,'-map','0:a:0','-ar','16000','-ac','1','-c:a','pcm_f32le',wave)
    audio,sr=sf.read(wave,dtype='float32')
    from funasr import AutoModel
    model=AutoModel(model=str(ROOT/'models/SenseVoiceSmall'),device='cuda:0',ncpu=4,disable_update=True,disable_pbar=True,trust_remote_code=False)
    rows=[]
    for row in alignment['rows']:
        if row['index'] not in indices:continue
        a=audio[round(row['window'][0]*sr):round(row['window'][1]*sr)]
        r=model.generate(input=a,fs=sr,language='zh',use_itn=True,disable_pbar=True)[0]
        raw=r.get('text','');plain=re.sub(r'<\|[^|>]+\|>','',raw).strip()
        result=dict(index=row['index'],expected=row['synth'],raw=raw,plain=plain,match=norm(plain)==norm(row['synth']))
        rows.append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
write(OUT/'final-aac-speech-check.json',dict(video=str(video),video_sha256=sha(video),indices=indices,selection='All revised closing cues, primary-ASR ambiguous body cue 10, and three lowest voice-to-bed RMS cues.',rows=rows,passed=all(x['match'] for x in rows),human_listening=False))
