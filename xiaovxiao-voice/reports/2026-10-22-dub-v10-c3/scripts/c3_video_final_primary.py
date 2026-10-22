"""Retain independent primary ASR on the final encoded closing, including disagreement."""
import tempfile
from pathlib import Path
import soundfile as sf
from c3_video_align import OUT, read, write, sha, norm, dub

report=read(OUT/'mix-report.json')
alignment=read(OUT/'aligned/alignment.json')
video=Path(report['outputs']['final_mp4'])
with tempfile.TemporaryDirectory(prefix='c3-final-primary-') as temp:
    wave=Path(temp)/'decoded.wav'
    dub.run_ffmpeg('-i',video,'-map','0:a:0','-ar','16000','-ac','1','-c:a','pcm_f32le',wave)
    audio,sr=sf.read(wave,dtype='float32')
    from funasr import AutoModel
    model=AutoModel(model='paraformer-zh',device='cuda:0',ncpu=4,disable_update=True,disable_pbar=True)
    rows=[]
    for row in alignment['rows'][-3:]:
        a=audio[round(row['window'][0]*sr):round(row['window'][1]*sr)]
        r=model.generate(input=a,fs=sr,disable_pbar=True)[0]
        result=dict(index=row['index'],expected=row['synth'],raw=r['text'],timestamp=r.get('timestamp'),match=norm(r['text'])==norm(row['synth']))
        rows.append(result)
        print(result,flush=True)
write(OUT/'final-aac-primary-closing.json',dict(video_sha256=sha(video),rows=rows,passed=all(x['match'] for x in rows),human_listening=False))
