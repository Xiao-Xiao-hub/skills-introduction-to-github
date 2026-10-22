"""One-off C3 brand-tag repair: only synthesize the final two syllables."""
from __future__ import annotations

import gc
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from common import VoxCPM2Backend
from c_consistency_select import normalize_transcript

OUT=ROOT/'dub/out/c3-video-2026-10-22/closing/brand-tag'
PLAN=ROOT/'reports/2026-10-22-dub-c3-expression/plan.json'
EXPECTED='晓念。'
SYNTHESIS_INPUT='小念。'
CUE=[175.35,176.25]
SEEDS=[89,97]

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(p,o):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def release(obj):
    del obj;gc.collect()
    try:
        import torch;torch.cuda.empty_cache()
    except Exception: pass

def plain(s): return re.sub(r'<\|[^|>]+\|>','',s or '').strip()

def match(s):
    # Existing documented phonetic mapping permits 小念 -> 晓念.  It never maps 想 -> 晓.
    return normalize_transcript(s)==normalize_transcript(EXPECTED)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if normalize_transcript(SYNTHESIS_INPUT)!=normalize_transcript(EXPECTED):raise SystemExit('alias does not normalize to expected')
    plan=json.loads(PLAN.read_text(encoding='utf-8')); anchor=plan['expression_bank']['warm']
    ref=ROOT/anchor['path']
    if sha(ref)!=anchor['sha256']:raise SystemExit('approved C3 warm reference hash mismatch')
    profile=next(x for x in plan['profiles'] if x['id']=='warm')
    rows=[];backend=VoxCPM2Backend(str(ROOT/'models/VoxCPM2'))
    try:
        for seed in SEEDS:
            path=OUT/f'c3_brand_tag_promptref_s{seed}.wav'
            # Prompt/reference continuation is intentionally mutually exclusive
            # with control text; no style instructions are passed here.
            wav,sr,used=backend._generate(seed,profile['gen'],text=SYNTHESIS_INPUT,prompt_wav_path=str(ref),prompt_text=anchor['text'],reference_wav_path=str(ref))
            sf.write(path,wav,sr,subtype='PCM_24')
            rows.append({'path':str(path.resolve()),'sha256':sha(path),'seed':seed,'seed_used':used,'mode':'prompt+reference_no_control','reference':str(ref.resolve()),'reference_sha256':anchor['sha256'],'prompt_text':anchor['text'],'original_text':EXPECTED,'synthesis_input':SYNTHESIS_INPUT,'duration':len(wav)/sr,'sample_rate':sr})
    finally:
        release(backend)
    from funasr import AutoModel
    para=AutoModel(model='paraformer-zh',device='cuda:0',ncpu=4,disable_update=True,disable_pbar=True)
    try:
        for row in rows:
            a,sr=sf.read(row['path'],dtype='float32');a16=resample_poly(a,16000,sr).astype(np.float32) if sr!=16000 else a
            r=para.generate(input=a16,fs=16000,batch_size=1,disable_pbar=True)[0]
            ts=r.get('timestamp',[]);span=(ts[-1][1]-ts[0][0])/1000 if len(ts)>=2 else None
            row['paraformer']={'text':r.get('text',''),'timestamp':ts,'strict_match':match(r.get('text',''))}
            row['cue_fit']={'window':CUE,'window_seconds':.9,'raw_speech_seconds':span,'needed_tempo':(span/.9 if span is not None else None),'passes_max_1_10':bool(span is not None and span<=.99)}
    finally:
        release(para)
    sense=AutoModel(model=str(ROOT/'models/SenseVoiceSmall'),device='cuda:0',ncpu=4,disable_update=True,disable_pbar=True,trust_remote_code=False)
    try:
        for row in rows:
            a,sr=sf.read(row['path'],dtype='float32');a16=resample_poly(a,16000,sr).astype(np.float32) if sr!=16000 else a
            r=sense.generate(input=a16,fs=16000,language='zh',use_itn=True,disable_pbar=True)[0]
            raw=r.get('text',''); row['sensevoice']={'raw':raw,'plain':plain(raw),'timestamp':r.get('timestamp'),'strict_match':match(plain(raw))}
    finally:
        release(sense)
    for row in rows:row['accepted']=row['paraformer']['strict_match'] and row['sensevoice']['strict_match'] and row['cue_fit']['passes_max_1_10']
    result={'original_text':EXPECTED,'synthesis_input':SYNTHESIS_INPUT,'alias_reason':'Use 小念 only as a same-pronunciation synthesis input; expected text remains 晓念 and 想 is never accepted as 晓.','reference_sha256':anchor['sha256'],'rows':rows,'accepted_paths':[r['path'] for r in rows if r['accepted']]}
    write(OUT/'brand-tag-results.json',result)
    accepted=[r for r in rows if r['accepted']]
    write(OUT/'selected-brand.json',({'passed':True,**accepted[0],'reason':'Passed both ASRs against original normalized text and the 0.9-second cue fit.'} if accepted else {'passed':False,'reason':'Neither of the two bounded candidates passed both ASRs and cue fit.','results_json':str((OUT/'brand-tag-results.json').resolve())}))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['accepted_paths'] else 1

if __name__=='__main__':raise SystemExit(main())
