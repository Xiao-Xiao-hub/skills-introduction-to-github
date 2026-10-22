"""Build an explicit candidate selection using unchanged prepare/assemble functions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import dub
from common import read_manifest

OUT=ROOT/'dub/out/c-consistent-2026-10-22'
REPORT=ROOT/'reports/2026-10-22-dub-c-consistent'

def read(name): return json.loads((OUT/name).read_text(encoding='utf-8'))
def write(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def normalize_transcript(text):
    # Only spelling variants with identical sounds, all surfaced in the report.
    import re
    value=re.sub(r'[^\u4e00-\u9fffA-Za-z]','',text).lower()
    for a,b in [('明瞭','明了'),('小念','晓念'),('小是让','晓是让'),('要做的是','要做的事'),('原画','原话'),('原化','原话'),('代办','待办'),('小黎','小狸'),('小离','小狸'),('他会','它会'),('在反馈','再反馈'),('先际状态','先记状态'),('算是另外','算式另外'),('sqlight','sqlite')]:
        value=value.replace(a,b)
    return value

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--assemble',action='store_true');args=parser.parse_args()
    plan=json.loads((REPORT/'plan.json').read_text(encoding='utf-8'))
    basic=read('basic-cache.json');speaker=read('speaker-cache.json');asr=read('asr-cache.json')
    profile=next(p for p in plan['profiles'] if p['id']==plan['selected_profile'])
    refsha=profile['sha256'];refvec=np.array(speaker[refsha]['embedding']);reff0=basic[refsha]['f0_median']
    if hashlib.sha256((ROOT/profile['reference']).read_bytes()).hexdigest()!=refsha:
        raise SystemExit('Fixed reference hash mismatch')
    data=dub.load_script(dub.SCRIPT);dub.flat_lines(data)
    entries=read_manifest(OUT/'takes')
    chosen={}; details=[];failed=[]
    for group in data['groups']:
        candidates=[]
        for e in entries:
            if e['group']!=group['id']:continue
            path=OUT/'takes'/e['file'];sha=hashlib.sha256(path.read_bytes()).hexdigest()
            if not all(sha in cache for cache in [basic,speaker,asr]):continue
            b=basic[sha];a=asr[sha];v=np.array(speaker[sha]['embedding'])
            audio,sr=sf.read(path,dtype='float64');prep=dub.prepare_take(audio,sr,group,1.05)
            sim=float(v@refvec);st=float(12*np.log2(b['f0_median']/reff0))
            cer=dub.char_error_rate(group['text'],a['text'])
            same=normalize_transcript(group['text'])==normalize_transcript(a['text'])
            checks={'fixed_reference':e['reference_sha256']==refsha,'same_text':e['text']==group['text'],
                    'final_cue_not_rejected':sha not in plan.get('rejected_take_sha256',{}),
                    'profile':e['profile']==profile['id'],'mode':e['mode']=='clone','style':e['style']=='',
                    'reference_text':e['reference_text']==profile['reference_text'],'gen':e['gen']==profile['gen'],
                    'sample_rate':sr==48000,'seed_recorded':e['seed']==e['seed_used'],
                    'source_not_fullscale':b.get('source_fullscale_samples',0)==0,
                    'speaker':sim>=.85,'pitch':abs(st)<=2,'hnr':b['acoustic']['hnr']>=10,
                    'rate':b['acoustic']['rate']<=6.5,'split':prep['split_ok'],
                    'duration':prep['need']<=1.05,'transcript':same}
            row={'group':group['id'],'file':e['file'],'sha256':sha,'take':e['take'],
                 'speaker_cosine':sim,'f0_median_semitones_from_reference':st,**b,
                 'split_ok':prep['split_ok'],'need':prep['need'],'asr':a['text'],'cer':cer,
                 'expected':group['text'],'checks':checks,'failed':[k for k,vv in checks.items() if not vv]}
            # Technical criteria are pass/fail; prefer original fixed seed once admitted.
            row['rank']=[len(row['failed']),0 if e['seed']==41 else 1,abs(st),-sim]
            candidates.append((row,dict(entry=e,sr=sr,prep=prep,cer=cer,score=0,penalty=0)))
        if not candidates:
            failed.append(group['id']);continue
        candidates.sort(key=lambda c:c[0]['rank'])
        best,c=candidates[0]
        chosen[group['id']]=c
        details.append({'group':group['id'],'selected':best,'candidates':[x[0] for x in candidates]})
        if best['failed']:failed.append(group['id'])
    write(OUT/'candidate-review.json',{'failed_groups':failed,'groups':details,'normalization':'Explicit identical-sound spelling variants only; raw transcripts retained.'})
    for d in details:
        s=d['selected'];print(s['group'],s['take'],f"sim={s['speaker_cosine']:.3f}",f"pitch={s['f0_median_semitones_from_reference']:+.2f}st",f"need={s['need']:.3f}",s['failed'],s['asr'] if not s['checks']['transcript'] else '',flush=True)
    print('RETRY',','.join(failed))
    if args.assemble:
        if failed:raise SystemExit('Candidate gates failed; no final audio written.')
        sr=next(iter(chosen.values()))['sr']
        track,placed,warnings=dub.assemble(data,chosen,sr,1.05)
        arg=SimpleNamespace(ref='C_沉稳一致_固定参考',mode='clone',video=None)
        dub.write_outputs(arg,data,track,sr,placed,warnings,OUT)
        write(OUT/'selection.json',{g:{'file':v['entry']['file'],'sha256':hashlib.sha256((OUT/'takes'/v['entry']['file']).read_bytes()).hexdigest(),'seed':v['entry']['seed'],'reference_sha256':refsha,'gen':v['entry']['gen']} for g,v in chosen.items()})

if __name__=='__main__':main()
