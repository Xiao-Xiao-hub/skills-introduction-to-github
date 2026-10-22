"""Independent recognizer on actual aligned cues; retain disagreements verbatim."""
import json
import re
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from c3_video_align import OUT, ROOT, read, write, sha, norm

folder=OUT/'aligned'
alignment=read(folder/'alignment.json')
wave=folder/'人声轨_48k24bit.wav'
assert sha(wave)==alignment['waveform_sha256']
audio,sr=sf.read(wave,dtype='float32')
from funasr import AutoModel
model=AutoModel(model=str(ROOT/'models/SenseVoiceSmall'),device='cuda:0',ncpu=4,disable_update=True,disable_pbar=True,trust_remote_code=False)
rows=[]
for row in alignment['rows']:
    if row['index'] not in (10,53,54,55):continue
    a=audio[round(row['start']*sr):round(row['end']*sr)]
    result=model.generate(input=resample_poly(a,16000,sr).astype(np.float32),fs=16000,language='zh',use_itn=True,disable_pbar=True)[0]
    raw=result.get('text','')
    plain=re.sub(r'<\|[^|>]+\|>','',raw).strip()
    r=dict(index=row['index'],expected=row['synth'],raw=raw,plain=plain,match=norm(plain)==norm(row['synth']))
    rows.append(r)
    print(json.dumps(r,ensure_ascii=False),flush=True)
write(folder/'secondary-cue-verification.json',dict(waveform_sha256=sha(wave),rows=rows,passed=all(r['match'] for r in rows),human_listening=False))
