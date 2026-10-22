"""Read-only audio checks with cache, separate from the unchanged dubbing scripts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dub
import measure
from common import read_manifest

OUT = ROOT/'dub/out/c-consistent-2026-10-22'
REPORT = ROOT/'reports/2026-10-22-dub-c-consistent'

def write(path, data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)

def fingerprint(kind):
    import importlib.metadata as metadata
    files=[ROOT/'measure.py'] if kind=='basic' else ([ROOT/'models/CAMPlusPlus/config.yaml',ROOT/'models/CAMPlusPlus/campplus_cn_common.bin'] if kind=='speaker' else list((ROOT/'.cache/modelscope').glob('**/speech_seaco_paraformer*/**/config.yaml')))
    if kind=='asr':
        files=list((ROOT/'.cache/modelscope').glob('**/config.yaml'))
        files=[f for f in files if 'seaco_paraformer' in str(f)]
        files += [f.parent/'model.pt' for f in files if (f.parent/'model.pt').exists()]
    return {'schema':1,'kind':kind,'funasr':metadata.version('funasr'),'torch':metadata.version('torch'),
            'fs':16000,'pitch_floor':100,'pitch_ceiling':500,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}

def records(folder):
    plan=json.loads((REPORT/'plan.json').read_text(encoding='utf-8'))
    rows=[]
    for p in plan['profiles']:
        rows.append(dict(path=p['reference'],text=p['reference_text'],set='reference',profile=p['id'],group='reference'))
    old=json.loads((ROOT/'reports/2026-10-22-dub-v10/C_沉稳讲述/report.json').read_text(encoding='utf-8'))
    meta={e['file']:e for e in read_manifest(ROOT/'dub/out/r0_vox_959950_style/takes')}
    for f in dict.fromkeys(p['file'] for p in old['lines']):
        e=meta[f]
        rows.append(dict(path=str(Path('dub/out/r0_vox_959950_style/takes')/f),text=e['text'],set='old_C',profile='old_C',group=e['group']))
    for ref in ['r0b_vox_49a14f','r0b_moss_611767','r0b_vox_821713']:
        p,e=dub.find_clip(ref)
        rows.append(dict(path=str(p),text=e['text'],set='other_reference',profile=ref,group='reference'))
    for e in read_manifest(OUT/folder):
        rows.append(dict(path=str(OUT/folder/e['file']),text=e['text'],set=folder,profile=e.get('profile','unknown'),group=e['group']))
    if folder=='audit-groups':
        for e in read_manifest(OUT/'old-final-groups'):
            rows.append(dict(path=str(OUT/'old-final-groups'/e['file']),text=e['text'],set='old-final-groups',profile='old_final',group=e['group']))
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=['basic','speaker','asr'])
    p.add_argument('--folder',default='pilot')
    p.add_argument('--only-current',action='store_true')
    args=p.parse_args()
    rows=records(args.folder)
    if args.only_current:
        rows=[r for r in rows if r['set']==args.folder]
    cache_path=OUT/(args.kind+'-cache.json')
    meta_path=OUT/(args.kind+'-cache.meta.json')
    fp=fingerprint(args.kind)
    if cache_path.exists() and (not meta_path.exists() or json.loads(meta_path.read_text(encoding='utf-8'))!=fp):
        cache_path.replace(cache_path.with_suffix('.before-fingerprint-'+str(time.time_ns())+'.json'))
    write(meta_path,fp)
    cache=json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else {}
    model=None
    if args.kind in ['speaker','asr']:
        from funasr import AutoModel
        if args.kind=='speaker':
            model=AutoModel(model=str(ROOT/'models/CAMPlusPlus'),device='cpu',ncpu=4,disable_update=True,disable_pbar=True)
        else:
            model=AutoModel(model='paraformer-zh',device='cuda:0',ncpu=4,disable_update=True,disable_pbar=True)
    results=[]
    for i,row in enumerate(rows,1):
        path=ROOT/row['path']
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        text_sha=hashlib.sha256(row['text'].encode()).hexdigest()
        if sha not in cache or (args.kind=='basic' and cache[sha].get('text_sha256')!=text_sha):
            audio,sr=sf.read(path,dtype='float32')
            if audio.ndim>1: audio=audio.mean(axis=1)
            if args.kind=='basic':
                sound=measure.parselmouth.Sound(audio,sampling_frequency=sr)
                pitch=sound.to_pitch_ac(time_step=.01,pitch_floor=100,pitch_ceiling=500)
                f0=pitch.selected_array['frequency']; f0=f0[f0>0]
                m=measure.analyze(path,row['text'])
                cache[sha]=dict(acoustic=m,f0_median=float(np.median(f0)),f0_p10=float(np.percentile(f0,10)),f0_p90=float(np.percentile(f0,90)),
                    peak_db=float(20*np.log10(max(np.abs(audio).max(),1e-12))),duration=len(audio)/sr,
                    speech_rms_db=float(dub.speech_rms_db(audio,sr)),text_sha256=text_sha,
                    source_fullscale_samples=int(np.count_nonzero(np.abs(audio)>=.999)))
            else:
                audio16=resample_poly(audio,16000,sr).astype(np.float32)
                t=time.time(); result=model.generate(input=audio16,fs=16000,batch_size=1,disable_pbar=True)[0]
                if args.kind=='speaker':
                    v=result['spk_embedding'].detach().cpu().numpy().reshape(-1)
                    v=v/np.linalg.norm(v)
                    cache[sha]={'embedding':v.tolist(),'dimensions':len(v)}
                else:
                    cache[sha]={k:v for k,v in result.items() if k in ['text','timestamp']}
                    cache[sha]['seconds']=time.time()-t
            write(cache_path,cache)
        result=dict(row,sha256=sha,**cache[sha])
        if args.kind=='asr':
            result['cer']=dub.char_error_rate(row['text'],result['text'])
            result['expected']=row['text']
        results.append(result)
        print(f"{args.kind} {i}/{len(rows)} {row['profile']} {row['group']}",flush=True)
    write(OUT/(args.folder+'-'+args.kind+'.json'),results)
    if args.kind=='speaker':
        summaries={}
        ref={r['profile']:np.array(r['embedding']) for r in results if r['set']=='reference'}
        for profile in dict.fromkeys(r['profile'] for r in results if r['set'] not in ['reference','other_reference']):
            sample=[r for r in results if r['profile']==profile and r['set']!='reference']
            v=np.array([r['embedding'] for r in sample]); pair=v@v.T
            pairs=pair[np.triu_indices(len(sample),1)]
            scores={name:(v@vec).tolist() for name,vec in ref.items()}
            summaries[profile]={'count':len(sample),'pairwise_min':float(pairs.min()) if len(pairs) else None,
                'pairwise_p10':float(np.percentile(pairs,10)) if len(pairs) else None,
                'pairwise_mean':float(pairs.mean()) if len(pairs) else None,
                'reference_scores':scores,'files':[r['path'] for r in sample]}
        write(OUT/(args.folder+'-speaker-summary.json'),summaries)
        print(json.dumps(summaries,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
