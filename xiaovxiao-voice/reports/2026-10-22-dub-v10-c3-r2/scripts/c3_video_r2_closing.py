"""Bounded R2 whole-closing prompt/reference candidates; no source script changes."""
from __future__ import annotations

import gc, hashlib, json, re, sys
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from common import VoxCPM2Backend
from c_consistency_select import normalize_transcript

OUT=ROOT/'dub/out/c3-video-r2-2026-10-22/closing'
PLAN=ROOT/'reports/2026-10-22-dub-c3-expression/plan.json'
ORIGINAL='念，是转瞬的念头。晓，是让所想渐渐明瞭。晓念。'
VARIANTS={
    'A':'念，是转瞬的念头；小，是让所想渐渐明瞭。小念。',
    'B':'念是转瞬的念头，小是让所想渐渐明瞭——小念。',
}
SEEDS=[103,107]

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,o):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def release(o):
    del o;gc.collect()
    try:
        import torch;torch.cuda.empty_cache()
    except Exception:pass
def plain(s):return re.sub(r'<\|[^|>]+\|>','',s or '').strip()
def match(s):return normalize_transcript(s)==normalize_transcript(ORIGINAL)
def pauses(ts):
    # Three original lines have 7, 9, and 2 Han characters.  This reports
    # recognizer timing only; it never edits or slices the generated take.
    if len(ts)<18:return {'available':False,'timestamp_count':len(ts)}
    return {'available':True,'after_line1_seconds':round((ts[7][0]-ts[6][1])/1000,3),'after_line2_seconds':round((ts[16][0]-ts[15][1])/1000,3),'timestamp_count':len(ts)}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if any(normalize_transcript(v)!=normalize_transcript(ORIGINAL) for v in VARIANTS.values()):raise SystemExit('alias variants must normalize to original')
    plan=json.loads(PLAN.read_text(encoding='utf-8')); anchor=plan['expression_bank']['warm'];ref=ROOT/anchor['path']
    if sha(ref)!=anchor['sha256']:raise SystemExit('approved warm reference hash mismatch')
    gen=next(p['gen'] for p in plan['profiles'] if p['id']=='warm')
    rows=[]; backend=VoxCPM2Backend(str(ROOT/'models/VoxCPM2'))
    try:
        for label,text in VARIANTS.items():
            for seed in SEEDS:
                path=OUT/f'r2_closing_{label}_s{seed}.wav'
                # High-fidelity continuation: prompt + reference, no control text.
                wav,sr,used=backend._generate(seed,gen,text=text,prompt_wav_path=str(ref),prompt_text=anchor['text'],reference_wav_path=str(ref))
                sf.write(path,wav,sr,subtype='PCM_24')
                rows.append({'id':path.stem,'path':str(path.resolve()),'sha256':sha(path),'variant':label,'seed':seed,'seed_used':used,'mode':'prompt+reference_no_control','reference':str(ref.resolve()),'reference_sha256':anchor['sha256'],'prompt_text':anchor['text'],'original_text':ORIGINAL,'synthesis_input':text,'duration':len(wav)/sr,'sample_rate':sr,'source_windows':[[169.75,171.85],[172.12,174.35],[175.35,176.25]]})
    finally:release(backend)
    from funasr import AutoModel
    para=AutoModel(model='paraformer-zh',device='cuda:0',ncpu=4,disable_update=True,disable_pbar=True)
    try:
        for row in rows:
            a,sr=sf.read(row['path'],dtype='float32');a16=resample_poly(a,16000,sr).astype(np.float32) if sr!=16000 else a
            r=para.generate(input=a16,fs=16000,batch_size=1,disable_pbar=True)[0];ts=r.get('timestamp',[])
            row['paraformer']={'text':r.get('text',''),'timestamp':ts,'strict_match':match(r.get('text',''))};row['group_pauses']=pauses(ts)
    finally:release(para)
    sense=AutoModel(model=str(ROOT/'models/SenseVoiceSmall'),device='cuda:0',ncpu=4,disable_update=True,disable_pbar=True,trust_remote_code=False)
    try:
        for row in rows:
            a,sr=sf.read(row['path'],dtype='float32');a16=resample_poly(a,16000,sr).astype(np.float32) if sr!=16000 else a
            r=sense.generate(input=a16,fs=16000,language='zh',use_itn=True,disable_pbar=True)[0];raw=r.get('text','')
            row['sensevoice']={'raw':raw,'plain':plain(raw),'timestamp':r.get('timestamp'),'strict_match':match(plain(raw))}
    finally:release(sense)
    for row in rows:row['dual_asr_match']=row['paraformer']['strict_match'] and row['sensevoice']['strict_match']
    result={'scope':'Four whole-group prompt/reference candidates only; no per-line cutting, brand-tag append, control text, or source-script edit.','normalization':'小念 and 小是让 are documented same-sound variants; 想 is never normalized to 晓.','reference_sha256':anchor['sha256'],'rows':rows}
    write(OUT/'r2-closing-results.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
