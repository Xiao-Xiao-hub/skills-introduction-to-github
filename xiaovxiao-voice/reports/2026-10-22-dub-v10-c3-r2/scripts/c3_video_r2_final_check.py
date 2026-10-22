"""Check encoded final audio and measure actual soundtrack change versus R1."""
import gc
import re
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from c3_video_align import ROOT, read, write, sha, norm, dub
from c3_video_r2_plan import OUT

mix=read(OUT/'mix-report.json');alignment=read(OUT/'aligned/alignment.json')
video=Path(mix['video']);assert sha(video)==mix['video_sha256']
groups=[g for g in alignment['groups']if g['group']in ('05-06','07','08','10','20','23b','26-27')]
from funasr import AutoModel
import torch
rows=[]
with tempfile.TemporaryDirectory(prefix='c3-r2-aac-')as temp:
    decoded=Path(temp)/'decoded.wav'
    dub.run_ffmpeg('-i',video,'-map','0:a:0','-ac',1,'-ar',16000,'-c:a','pcm_f32le',decoded)
    audio,sr=sf.read(decoded,dtype='float32')
    model=AutoModel(model=str(ROOT/'models/SenseVoiceSmall'),device='cuda:0',ncpu=4,disable_update=True,disable_pbar=True,trust_remote_code=False)
    for g in groups:
        a=audio[round(g['start']*sr):round(g['end']*sr)]
        expected=''.join(l['synth']for l in g['lines'])
        r=model.generate(input=a,fs=sr,language='zh',use_itn=True,disable_pbar=True)[0]
        plain=re.sub(r'<\|[^|>]+\|>','',r.get('text','')).strip()
        row=dict(group=g['group'],expected=expected,sensevoice=dict(raw=r.get('text',''),plain=plain,match=norm(plain)==norm(expected)))
        rows.append(row);print(g['group'],plain,row['sensevoice']['match'],flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
    failed=[r for r in rows if not r['sensevoice']['match']]
    if failed:
        model=AutoModel(model='paraformer-zh',device='cuda:0',ncpu=4,disable_update=True,disable_pbar=True)
        for row in failed:
            g=next(g for g in groups if g['group']==row['group'])
            r=model.generate(input=audio[round(g['start']*sr):round(g['end']*sr)],fs=sr,disable_pbar=True)[0]
            row['paraformer']=dict(raw=r['text'],match=norm(r['text'])==norm(row['expected']))
            print('cross-check',row['group'],row['paraformer'],flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
old,sr=sf.read(ROOT/'dub/out/c3-video-2026-10-22/mix.wav',dtype='float32')
new,sr2=sf.read(OUT/'mix.wav',dtype='float32');assert sr==sr2==48000
metrics=[]
for name,start,end in [('body',14.18,167.88),('closing',169.5,176.25)]:
    a,b=round(start*sr),round(end*sr)
    # Centered mono voice cancels; the side component directly measures audible
    # stereo background in the actual mixed PCM, without estimating voice gain.
    old_side=(old[a:b,0]-old[a:b,1])/2;new_side=(new[a:b,0]-new[a:b,1])/2
    old_db=float(20*np.log10(np.sqrt(np.mean(old_side.astype(np.float64)**2))+1e-12))
    new_db=float(20*np.log10(np.sqrt(np.mean(new_side.astype(np.float64)**2))+1e-12))
    metrics.append(dict(section=name,old_side_rms_dbfs=old_db,new_side_rms_dbfs=new_db,increase_db=new_db-old_db))
passed=all(r['sensevoice']['match']or r.get('paraformer',{}).get('match',False)for r in rows)
write(OUT/'final-aac-check.json',dict(video_sha256=sha(video),rows=rows,cross_checked_passed=passed,background_comparison=metrics,
    background_metric_scope='Actual stereo side component of final mixed PCM; centered narration cancels. This measures a background component, not subjective music loudness.',human_listening=False))
print('RESULT',{'passed':passed,'background_comparison':metrics},flush=True)
