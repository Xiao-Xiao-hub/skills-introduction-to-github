"""Final bounded plain-character R2 whole-closing candidates only."""
from __future__ import annotations
import gc,hashlib,json,re,sys
from pathlib import Path
import soundfile as sf
from scipy.signal import resample_poly

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from common import VoxCPM2Backend
from c_consistency_select import normalize_transcript
OUT=ROOT/'dub/out/c3-video-r2-2026-10-22/closing'; PLAN=ROOT/'reports/2026-10-22-dub-c3-expression/plan.json'
ORIGINAL='念，是转瞬的念头。晓，是让所想渐渐明瞭。晓念。'
INPUT='念，是转瞬的念头；小，是让所想渐渐明了。小念。'
SEEDS=[109,113]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def release(x):
 del x;gc.collect()
 try:
  import torch;torch.cuda.empty_cache()
 except Exception:pass
def plain(s):return re.sub(r'<\|[^|>]+\|>','',s or '').strip()
def match(s):return normalize_transcript(s)==normalize_transcript(ORIGINAL)
def pauses(ts):
 if len(ts)<18:return {'available':False,'timestamp_count':len(ts)}
 return {'available':True,'after_line1_seconds':round((ts[7][0]-ts[6][1])/1000,3),'after_line2_seconds':round((ts[16][0]-ts[15][1])/1000,3),'timestamp_count':len(ts)}
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if normalize_transcript(INPUT)!=normalize_transcript(ORIGINAL):raise SystemExit('input must normalize to original')
 plan=json.loads(PLAN.read_text(encoding='utf-8'));anchor=plan['expression_bank']['warm'];ref=ROOT/anchor['path']
 if sha(ref)!=anchor['sha256']:raise SystemExit('approved reference changed')
 gen=next(p['gen'] for p in plan['profiles'] if p['id']=='warm');rows=[];b=VoxCPM2Backend(str(ROOT/'models/VoxCPM2'))
 try:
  for seed in SEEDS:
   p=OUT/f'r2_plaintext_s{seed}.wav';wav,sr,used=b._generate(seed,gen,text=INPUT,prompt_wav_path=str(ref),prompt_text=anchor['text'],reference_wav_path=str(ref));sf.write(p,wav,sr,subtype='PCM_24')
   rows.append({'id':p.stem,'path':str(p.resolve()),'sha256':sha(p),'seed':seed,'seed_used':used,'mode':'prompt+reference_no_control','reference':str(ref.resolve()),'reference_sha256':anchor['sha256'],'prompt_text':anchor['text'],'original_text':ORIGINAL,'synthesis_input':INPUT,'duration':len(wav)/sr,'sample_rate':sr})
 finally:release(b)
 from funasr import AutoModel
 m=AutoModel(model='paraformer-zh',device='cuda:0',ncpu=4,disable_update=True,disable_pbar=True)
 try:
  for x in rows:
   a,sr=sf.read(x['path'],dtype='float32');a=resample_poly(a,16000,sr).astype('float32') if sr!=16000 else a;r=m.generate(input=a,fs=16000,batch_size=1,disable_pbar=True)[0];ts=r.get('timestamp',[]);x['paraformer']={'text':r.get('text',''),'timestamp':ts,'strict_match':match(r.get('text',''))};x['group_pauses']=pauses(ts)
 finally:release(m)
 m=AutoModel(model=str(ROOT/'models/SenseVoiceSmall'),device='cuda:0',ncpu=4,disable_update=True,disable_pbar=True,trust_remote_code=False)
 try:
  for x in rows:
   a,sr=sf.read(x['path'],dtype='float32');a=resample_poly(a,16000,sr).astype('float32') if sr!=16000 else a;r=m.generate(input=a,fs=16000,language='zh',use_itn=True,disable_pbar=True)[0];raw=r.get('text','');x['sensevoice']={'raw':raw,'plain':plain(raw),'timestamp':r.get('timestamp'),'strict_match':match(plain(raw))};x['dual_asr_match']=x['paraformer']['strict_match'] and x['sensevoice']['strict_match']
 finally:release(m)
 result={'scope':'Exactly two complete whole-group candidates; no cutting, inserted brand tag, or control text.','normalization':'小念 and 小是让 plus 明了/明瞭 are existing same-sound/spelling variants; 想 is never normalized to 晓.','rows':rows};write(OUT/'r2-plaintext-results.json',result);print(json.dumps(result,ensure_ascii=False,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
