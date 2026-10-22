"""Check the delivered waveform, not only the unedited source takes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import dub
from c_consistency_select import normalize_transcript

OUT=ROOT/'dub/out/c-consistent-2026-10-22'
REPORT=ROOT/'reports/2026-10-22-dub-c-consistent'

def write(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--extract-only',action='store_true');args=parser.parse_args()
    plan=json.loads((REPORT/'plan.json').read_text(encoding='utf-8'))
    rep=json.loads((OUT/'final/report.json').read_text(encoding='utf-8'))
    script=dub.load_script(dub.SCRIPT);dub.flat_lines(script)
    audio,sr=sf.read(OUT/'final/人声轨_48k24bit.wav',dtype='float32')
    info=sf.info(OUT/'final/人声轨_48k24bit.wav')
    expected=[l for g in script['groups'] for l in g['lines']]
    placed=rep['lines']
    if len(expected)!=55 or len(placed)!=55:raise SystemExit('Expected exactly 55 cues')
    cues=OUT/'audit-cues';cues.mkdir(exist_ok=True)
    groups=OUT/'audit-groups';groups.mkdir(exist_ok=True)
    cue_rows=[];group_rows=[];fail=[];timestamp_checks=[]
    for i,(l,p) in enumerate(zip(expected,placed)):
        if l['text']!=p['text']:raise SystemExit('Cue text/order mismatch')
        previous=placed[i-1]['end'] if i else 0
        next_start=placed[i+1]['start'] if i+1<len(placed) else 178
        s=max(0,p['start']-.04,previous+.005 if i else 0)
        e=min(178,p['end']+.04,next_start-.005)
        piece=audio[int(round(s*sr)):int(round(e*sr))]
        name=f"cue_{i+1:02d}_g{p['group']}.wav";path=cues/name
        sf.write(path,piece,sr,subtype='PCM_24')
        checks={'same_text':l['text']==p['text'],'split_ok':p['split_ok'],
                'start_delay':p['start']-l['start']<=.05+1e-6,
                'end_limit':p['end']<=l['end']+.35+.015,
                'gap':p['end']<=next_start-.06+.015 if i+1<len(placed) else p['end']<=178,
                'speed':p['speed']<=1.05+.001,'non_silent':float(np.max(np.abs(piece)))>.01,
                'peak_within_limit':float(np.max(np.abs(piece)))<.891}
        row={'id':path.stem,'file':name,'group':p['group'],'index':i+1,'text':l['synth'],
             'original_text':l['text'],'sr':sr,'duration':len(piece)/sr,'profile':'final',
             'placed_start':p['start'],'placed_end':p['end'],'extracted_start':s,'extracted_end':e,
             'window':p['window'],'speed':p['speed'],'checks':checks}
        cue_rows.append(row)
        for k,v in checks.items():
            if not v:fail.append({'cue':i+1,'group':p['group'],'check':k})
    for g in script['groups']:
        lines=[p for p in placed if p['group']==g['id']]
        s=max(0,lines[0]['start']-.02);e=min(178,lines[-1]['end']+.03)
        piece=audio[int(round(s*sr)):int(round(e*sr))]
        name=f"final_g{g['id']}.wav";sf.write(groups/name,piece,sr,subtype='PCM_24')
        group_rows.append({'id':Path(name).stem,'file':name,'group':g['id'],'profile':'final','text':g['text'],'sr':sr,'duration':len(piece)/sr})
    for folder,rows in [(cues,cue_rows),(groups,group_rows)]:
        (folder/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    if args.extract_only:return
    from funasr import AutoModel
    model=AutoModel(model='paraformer-zh',device='cuda:0',ncpu=4,disable_update=True,disable_pbar=True)
    from c_consistency_check import fingerprint
    fp=fingerprint('asr');mp=OUT/'final-cue-asr-cache.meta.json';cp=OUT/'final-cue-asr-cache.json'
    if cp.exists() and (not mp.exists() or json.loads(mp.read_text(encoding='utf-8'))!=fp):cp.replace(cp.with_suffix('.old-'+str(time.time_ns())+'.json'))
    write(mp,fp);cache=json.loads(cp.read_text(encoding='utf-8')) if cp.exists() else {}
    for i,row in enumerate(cue_rows,1):
        path=cues/row['file'];sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if sha not in cache:
            a,s=sf.read(path,dtype='float32');a16=resample_poly(a,16000,s).astype(np.float32)
            result=model.generate(input=a16,fs=16000,batch_size=1,disable_pbar=True)[0]
            cache[sha]={k:v for k,v in result.items() if k in ['text','timestamp']};write(cp,cache)
        row['asr']=cache[sha]
        row['raw_cer']=dub.char_error_rate(row['text'],row['asr']['text'])
        row['transcript_equal_after_spelling_normalization']=normalize_transcript(row['text'])==normalize_transcript(row['asr']['text'])
        if not row['transcript_equal_after_spelling_normalization']:
            fail.append({'cue':i,'group':row['group'],'check':'transcript','expected':row['text'],'recognized':row['asr']['text']})
        print(f"cue {i}/55 {'OK' if row['transcript_equal_after_spelling_normalization'] else 'REVIEW'} {row['text']} | {row['asr']['text']}",flush=True)
    source={name:{'expected':sha,'current':hashlib.sha256((ROOT/name).read_bytes()).hexdigest()} for name,sha in plan['source_hashes'].items()}
    all_sources_unchanged=all(v['expected']==v['current'] for v in source.values())
    if not all_sources_unchanged:fail.append({'check':'source_hashes_changed'})
    if sr!=48000 or info.channels!=1 or info.subtype!='PCM_24' or abs(len(audio)/sr-178)>1/sr:
        fail.append({'check':'output_format'})
    summary={'waveform_sha256':hashlib.sha256((OUT/'final/人声轨_48k24bit.wav').read_bytes()).hexdigest(),
             'duration':len(audio)/sr,'sample_rate':sr,'channels':info.channels,'subtype':info.subtype,
             'cue_count':len(cue_rows),'group_count':len(group_rows),'peak_dbfs':float(20*np.log10(np.max(np.abs(audio)))),
             'source_unchanged':all_sources_unchanged,'source_hashes':source,'checks_failed':fail,'cues':cue_rows,
             'content_timing_checks_passed':not fail,'final_acceptance':'Pending separate final consistency gate',
             'human_listening_status':'Not performed by a human reviewer; user final listening remains required.'}
    write(OUT/'final-waveform-verification.json',summary)
    print('FAILED_CHECKS',json.dumps(fail,ensure_ascii=False))
    if fail:raise SystemExit(1)

if __name__=='__main__':main()
