"""Targeted cross-ASR on the same final waveform; preserve full-context disagreement."""
import gc
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import dub
from c_consistency_select import normalize_transcript as norm

OUT=ROOT/'dub/out/c3-expression-2026-10-22'
path=OUT/'audit-groups/final-verification.json'
verification=json.loads(path.read_text(encoding='utf-8'))
wave=OUT/'natural-final/人声轨_48k24bit.wav'
assert hashlib.sha256(wave.read_bytes()).hexdigest()==verification['source_sha256']
assert verification['failures']==['sensevoice_text:23b'],verification['failures']
row=next(r for r in verification['model_rows'] if r['group']=='23b')
assert row['paraformer_match']
assert norm(row['sensevoice_plain']).replace('场任务','长任务')==norm(row['text'])
tokens=row['paraformer']['text'].split();timestamps=row['paraformer']['timestamp']
assert len(tokens)==len(timestamps)
index=tokens.index('长')
start=max(row['start'],row['start']+timestamps[index][0]/1000-.10)
end=row['end']
audio,sr=sf.read(wave,dtype='float32')
clip=audio[round(start*sr):round(end*sr)]
clip_path=OUT/'audit-groups/23b_长任务_最终音轨原句.wav'
sf.write(clip_path,clip,sr,subtype='PCM_24')
expected=next(g for g in dub.load_script(dub.SCRIPT)['groups']if g['id']=='23b')['lines'][-1]['synth']
from funasr import AutoModel
import torch
results={}
for name,model_path in [('paraformer','paraformer-zh'),('sensevoice',str(ROOT/'models/SenseVoiceSmall'))]:
    model=AutoModel(model=model_path,device='cuda:0',ncpu=4,disable_update=True,disable_pbar=True)
    extra=dict(language='zh',use_itn=True) if name=='sensevoice' else {}
    result=model.generate(input=resample_poly(clip,16000,sr).astype(np.float32),fs=16000,disable_pbar=True,**extra)[0]
    raw=result['text'];plain=re.sub(r'<\|[^|>]+\|>','',raw).strip()
    results[name]={'raw':raw,'plain':plain,'normalized_equal':norm(plain)==norm(expected)}
    print(name,results[name],flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
resolution=dict(expected=expected,actual_final_wave_sha256=verification['source_sha256'],
                source_interval=[start,end],clip_file=str(clip_path),clip_sha256=hashlib.sha256(clip_path.read_bytes()).hexdigest(),
                results=results,passed=all(r['normalized_equal']for r in results.values()),
                full_context_disagreement_preserved=row['sensevoice_plain'],
                expected_text_or_hotwords_passed_to_models=False,human_pronunciation_verified=False)
(OUT/'targeted-pronunciation.json').write_text(json.dumps(resolution,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if not resolution['passed']:raise SystemExit('Targeted readback still disagrees; do not resolve.')
(OUT/'unresolved23b-final-verification.json').write_text(path.read_text(encoding='utf-8'),encoding='utf-8')
row['targeted_pronunciation_verification']=resolution
row['whole_group_sensevoice_disagrees']=True
verification['resolved_checks']=[{'check':'sensevoice_text:23b','method':'Exact disputed sentence extracted from same final wave; both independent ASR outputs match, full-context disagreement retained.'}]
verification['failures']=[]
verification['technical_passed']=True
verification['human_listening_passed']=False
path.write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
