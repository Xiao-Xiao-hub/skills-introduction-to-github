import gc,hashlib,json,sys,re
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import dub
from common import read_manifest
from funasr import AutoModel
import torch
OUT=ROOT/'dub/out/c-consistent-2026-10-22';dst=OUT/'cross-asr';dst.mkdir(exist_ok=True)
data=dub.load_script(dub.SCRIPT);dub.flat_lines(data);groups={g['id']:g for g in data['groups']}
rows=[]
for e in read_manifest(OUT/'takes'):
    if e['group'] not in ['07','23b']:continue
    g=groups[e['group']]
    a,sr=sf.read(OUT/'takes'/e['file'],dtype='float64');prep=dub.prepare_take(a,sr,g,1.05)
    gain=10**((-20-dub.speech_rms_db(np.concatenate(prep['pieces']),sr))/20)
    if not prep['split_ok']:continue
    indices=[1] if g['id']=='07' else list(range(len(g['lines'])))
    for index in indices:
        p=dub.fade(dub.atempo(prep['pieces'][index]*gain,sr,prep['factor']),sr)
        if np.abs(p).max()>.89:p*=.89/np.abs(p).max()
        p=np.pad(p,(round(.04*sr),round(.04*sr)))
        name=Path(e['file']).stem+f'_cue{index+1}.wav';sf.write(dst/name,p,sr,subtype='PCM_24')
        rows.append({'file':str(dst/name),'source_file':e['file'],'group':g['id'],'cue_in_group':index+1,'seed':e['seed'],'expected':g['lines'][index]['synth']})
for e in read_manifest(OUT/'audit-cues'):
    if e['index'] in [5,7,11,31,32,34,43,44,47,48,53,54,55]:
        rows.append({'file':str(OUT/'audit-cues'/e['file']),'cue':e['index'],'expected':e['text']})
for model_name,path in [('paraformer','paraformer-zh'),('sensevoice',str(ROOT/'models/SenseVoiceSmall'))]:
    model=AutoModel(model=path,device='cuda:0',ncpu=4,disable_update=True,disable_pbar=True,trust_remote_code=False)
    for row in rows:
        a,sr=sf.read(row['file'],dtype='float32');a16=resample_poly(a,16000,sr).astype(np.float32)
        options={'language':'zh','use_itn':True} if model_name=='sensevoice' else {}
        result=model.generate(input=a16,fs=16000,batch_size=1,disable_pbar=True,**options)[0]
        row[model_name]={'raw_text':result['text'],'text':re.sub(r'<\|[^|]+\|>','',result['text']).strip()}
        print(model_name,row.get('group','final'),row.get('seed',row.get('cue')),row.get('cue_in_group',''),row['expected'],'|',row[model_name]['text'],flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
fingerprints={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'models/SenseVoiceSmall/model.pt',ROOT/'models/SenseVoiceSmall/config.yaml']}
(OUT/'cross-asr.json').write_text(json.dumps({'scope':'Generated candidate cue 10 plus key terms and spelling-variant cues; no hotword prompt provided.','sensevoice_model_sha256':fingerprints,'rows':rows},ensure_ascii=False,indent=2),encoding='utf-8')
