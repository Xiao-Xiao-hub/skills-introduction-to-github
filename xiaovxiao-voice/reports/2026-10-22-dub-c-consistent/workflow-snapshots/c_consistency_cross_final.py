import hashlib,json,re,sys
from pathlib import Path
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(Path(__file__).resolve().parent))
from c_consistency_select import normalize_transcript
from funasr import AutoModel
OUT=ROOT/'dub/out/c-consistent-2026-10-22'
verification=json.loads((OUT/'final-waveform-verification.json').read_text(encoding='utf-8'))
model=AutoModel(model=str(ROOT/'models/SenseVoiceSmall'),device='cuda:0',ncpu=4,disable_update=True,disable_pbar=True,trust_remote_code=False)
rows=[]
for row in verification['cues']:
    if row['index'] not in [5,7,10,11,31,32,34,43,44,47,48,53,54,55]:continue
    path=OUT/'audit-cues'/row['file'];audio,sr=sf.read(path,dtype='float32')
    result=model.generate(input=resample_poly(audio,16000,sr).astype(np.float32),fs=16000,language='zh',use_itn=True,disable_pbar=True)[0]
    text=re.sub(r'<\|[^|]+\|>','',result['text']).strip()
    equal=normalize_transcript(text)==normalize_transcript(row['text'])
    rows.append({'cue':row['index'],'group':row['group'],'expected':row['text'],'sensevoice_raw':result['text'],
        'sensevoice_text':text,'paraformer_text':row['asr']['text'],'same_after_spelling_normalization':equal,
        'cue_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    print(row['index'],equal,row['text'],'|',text,flush=True)
result={'all_key_cues_passed':all(r['same_after_spelling_normalization'] for r in rows),'model':'iic/SenseVoiceSmall',
        'waveform_sha256':verification['waveform_sha256'],'hotwords_used':False,'expected_text_supplied_to_model':False,
        'model_sha256':hashlib.sha256((ROOT/'models/SenseVoiceSmall/model.pt').read_bytes()).hexdigest(),'rows':rows}
(OUT/'cross-final-asr.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
if not result['all_key_cues_passed']:raise SystemExit(1)
