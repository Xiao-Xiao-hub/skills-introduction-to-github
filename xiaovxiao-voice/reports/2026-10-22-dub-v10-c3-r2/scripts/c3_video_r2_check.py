"""Transcribe the intact final phrase waveforms, with disagreements kept visible."""
import gc
import re
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from c3_video_align import ROOT, SR, read, write, sha, norm, canonical_units
from c3_video_r2_plan import OUT

folder=OUT/'aligned';alignment=read(folder/'alignment.json')
assert sha(folder/'人声轨_48k24bit.wav')==alignment['waveform_sha256']
from funasr import AutoModel
import torch
cache_path=OUT/'clean-group-asr.json'
cache={r['sha256']:r for r in read(cache_path)['rows']} if cache_path.exists() else {}
model=None
rows=[]
for g in alignment['groups']:
    path=g['rendered_path'];assert sha(path)==g['rendered_sha256']
    expected=''.join(l['synth']for l in g['lines'])
    if sha(path) in cache and cache[sha(path)]['expected']==expected:
        row={**cache[sha(path)],'start':g['start'],'end':g['end']}
        rows.append(row)
        print('cached',g['group'],row['primary_match'],flush=True)
        continue
    a,sr=sf.read(path,dtype='float32');assert sr==SR
    if model is None:model=AutoModel(model='paraformer-zh',device='cuda:0',ncpu=4,disable_update=True,disable_pbar=True)
    r=model.generate(input=resample_poly(a,16000,sr).astype(np.float32),fs=16000,disable_pbar=True)[0]
    row=dict(group=g['group'],path=path,sha256=sha(path),expected=expected,primary={k:r[k]for k in ['text','timestamp']},primary_match=norm(r['text'])==norm(expected),start=g['start'],end=g['end'])
    rows.append(row);print('primary',g['group'],r['text'],row['primary_match'],flush=True)
    write(OUT/'clean-group-asr.json',dict(waveform_sha256=alignment['waveform_sha256'],rows=rows,status='primary_in_progress'))
del model;gc.collect();torch.cuda.empty_cache()
needs=[r for r in rows if (not r['primary_match'] or r['group']=='26-27') and 'secondary' not in r]
if needs:
    model=AutoModel(model=str(ROOT/'models/SenseVoiceSmall'),device='cuda:0',ncpu=4,disable_update=True,disable_pbar=True,trust_remote_code=False)
    for row in needs:
        a,sr=sf.read(row['path'],dtype='float32')
        r=model.generate(input=resample_poly(a,16000,sr).astype(np.float32),fs=16000,language='zh',use_itn=True,disable_pbar=True)[0]
        plain=re.sub(r'<\|[^|>]+\|>','',r.get('text','')).strip()
        row['secondary']=dict(raw=r.get('text',''),plain=plain,match=norm(plain)==norm(row['expected']))
        print('secondary',row['group'],plain,row['secondary']['match'],flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
all_cues=[];gaps=[];unresolved=[];index=0
for g,row in zip(alignment['groups'],rows):
    row['cross_checked_match']=row['primary_match'] or row.get('secondary',{}).get('match',False)
    if not row['cross_checked_match']:unresolved.append(row['group'])
    canonical,spans=canonical_units(row['primary'])
    if len(canonical)!=len(norm(row['expected'])):
        row['timestamps_mapped']=False;continue
    row['timestamps_mapped']=True;offset=0;previous_end=None
    for i,line in enumerate(g['lines']):
        count=len(norm(line['synth']));speech_start=row['start']+spans[offset][0];speech_end=row['start']+spans[offset+count-1][1];offset+=count
        index+=1
        cue=dict(index=index,group=g['group'],text=line['text'],subtitle_window=[line['start'],line['end']],actual_asr_speech=[speech_start,speech_end],
            early_seconds=max(0,line['start']-speech_start),late_seconds=max(0,speech_end-line['end']))
        all_cues.append(cue)
        if previous_end is not None and g['group']!='26-27':
            gaps.append(dict(group=g['group'],before_index=index,gap_seconds=speech_start-previous_end))
        previous_end=speech_end
result=dict(waveform_sha256=alignment['waveform_sha256'],rows=rows,unresolved=unresolved,cross_checked_passed=not unresolved,
    primary_matches=sum(r['primary_match']for r in rows),secondary_used=[r['group']for r in rows if 'secondary' in r],cue_timing=all_cues,
    body_internal_gaps=gaps,mean_body_internal_gap_seconds=float(np.mean([g['gap_seconds']for g in gaps])),
    note='Whole-group ASR verifies edited waveform, not human naturalness. Position-only timestamp mapping for equal-length disagreements is disclosed; transcript mismatches remain raw.',human_listening=False)
write(OUT/'clean-group-asr.json',result)
print('RESULT',{'unresolved':unresolved,'primary_matches':result['primary_matches'],'mean_body_internal_gap':result['mean_body_internal_gap_seconds']},flush=True)
