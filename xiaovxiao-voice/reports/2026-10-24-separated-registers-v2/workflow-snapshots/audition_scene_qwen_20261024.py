"""Same-sentence register comparison plus new formal/colloquial utterances."""
from pathlib import Path
import sys,time,json
import numpy as np,soundfile as sf
import scene_voice_runtime_20261024 as R
T=R.T;S=R.S;OUT=S.BASE/'audition';OUT.mkdir(exist_ok=True)
SAME='我刚刚才看到消息，等一下就过来。你先别着急，我把这边弄好就来找你。'
TEXTS={
 'formal':{'intro':'本作品支持记录与整理。原始内容会一并保留，方便之后查看和继续思考。','same':SAME},
 'daily':{'reply':'要得，你先过去嘛。我这边还没弄好，等一下就来。你到了喊我一声。','grumble':'哎呀，我刚刚才晓得。我还以为是我弄错了，白整了半天，真是。','happy':'这个好乖哟！我也想拿来耍一下，感觉还挺安逸的。','same':SAME}}

def synth():
    for scene in ('formal','daily'):R.package(scene)
    manifest=OUT/'manifest.jsonl';done={r['id']for r in T.rows(manifest)};v=R.create_model(T.ROOT/'models/Qwen3-TTS-Xuan-formal-v2')
    for scene in ('formal','daily'):
        pack=T.ROOT/f'models/Qwen3-TTS-Xuan-{scene}-v2';meta=R.select(v,pack)
        for kind,text in TEXTS[scene].items():
            key=f'{scene}_{kind}'
            if key in done:continue
            T.seed(201);start=time.monotonic();wavs,sr=v.generate_custom_voice(text=text,language=meta['language'],speaker=meta['speaker'],temperature=.8,top_p=.9,max_new_tokens=640);p=OUT/(key+'.wav');sf.write(p,wavs[0],sr,subtype='PCM_24');T.append(manifest,dict(id=key,scene=scene,kind=kind,mode=scene,text=text,path=str(p),sr=sr,duration=len(wavs[0])/sr,seconds=time.monotonic()-start,sha256=T.sha(p),adapter_sha256=meta['adapter_sha256'],language=meta['language'],speaker=meta['speaker'],seed=201,reference_audio_required_at_inference=False,human_accepted=False));print('SYNTH',key,round(time.monotonic()-start,1),len(wavs[0])/sr,flush=True)

def evaluate():
    import qq_lora_audition_20261023 as A
    A.OUT=S.PREV;A.SAMPLES=OUT;A.evaluate()

def qwen_check():
    import torch
    from transformers import AutoModelForMultimodalLM,AutoProcessor
    from scipy.signal import resample_poly
    from opencc import OpenCC
    from rapidfuzz.distance import Levenshtein
    path=str(T.ROOT/'models/Qwen3-ASR-1.7B-hf');p=AutoProcessor.from_pretrained(path,local_files_only=True);m=AutoModelForMultimodalLM.from_pretrained(path,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval();conv=OpenCC('t2s');data=T.read(OUT/'evaluation.json')
    for r in data['rows']:
        a,sr=sf.read(r['path'],dtype='float32');a=resample_poly(a,16000,sr).astype(np.float32);args=dict(audio=[a],language=None,audio_kwargs={'sampling_rate':16000})
        if r['scene']=='daily':args['prompt']=[S.PROMPT]
        inp=p.apply_transcription_request(**args).to('cuda',dtype=torch.bfloat16)
        with torch.inference_mode():tokens=m.generate(**inp,max_new_tokens=256,do_sample=False)
        text=conv.convert(p.decode(tokens[:,inp['input_ids'].shape[1]:],return_format='transcription_only')[0]);r['qwen']=dict(raw=text,match=T.D.norm(text)==T.D.norm(r['text']),cer=Levenshtein.distance(T.D.norm(text),T.D.norm(r['text']))/len(T.D.norm(r['text'])),context_prompted=r['scene']=='daily');r['exact_matches']=sum(r[n]['match']for n in ('para','sense','qwen'));T.write(OUT/'evaluation.json',data);print('CHECK',r['id'],r['exact_matches'],flush=True)

def reference_check():
    from scipy.signal import resample_poly
    import qq_voice_corpus_20261023 as C
    m=C.model(T.ROOT/'models/CAMPlusPlus',device='cpu')
    def emb(path):
        a,sr=sf.read(path,dtype='float32');a=resample_poly(a,16000,sr).astype(np.float32);x=m.generate(input=a,fs=16000,disable_pbar=True)[0]['spk_embedding'].detach().cpu().numpy().ravel();return x/np.linalg.norm(x)
    anchors={scene:emb(S.BASE/scene/'reference24.wav')for scene in ('formal','daily')};data=T.read(OUT/'evaluation.json')
    for r in data['rows']:
        v=emb(r['path']);r['scene_reference_cosines']={scene:float(v@x)for scene,x in anchors.items()}
    data['formal_daily_reference_cosine']=float(anchors['formal']@anchors['daily']);data['similarities_are_not_identity_acceptance']=True;T.write(OUT/'evaluation.json',data);print('REFERENCE_CHECK',json.dumps([{k:r[k]for k in ('id','scene_reference_cosines')}for r in data['rows']]),flush=True)

def reload():
    pack=T.ROOT/'models/Qwen3-TTS-Xuan-daily-v2';v=R.create_model(pack);meta=R.select(v,pack);T.seed(201)
    wavs,sr=v.generate_custom_voice(text=SAME,language=meta['language'],speaker=meta['speaker'],temperature=.8,top_p=.9,max_new_tokens=640)
    p=OUT/'daily_same_fresh_reload.wav';sf.write(p,wavs[0],sr,subtype='PCM_24');a,fs=sf.read(OUT/'daily_same.wav',dtype='float32');b,bs=sf.read(p,dtype='float32');equal=fs==bs and np.array_equal(a,b)
    proof=dict(fresh_process=True,scene='daily',reference_audio_required=False,adapter_sha256=meta['adapter_sha256'],speaker_embedding_sha256=meta['speaker_embedding_sha256'],waveform_bit_identical=equal,sample_counts=[len(a),len(b)],max_abs_delta=float(np.max(np.abs(a-b)))if a.shape==b.shape else None)
    T.write(OUT/'adapter-reload-proof.json',proof);print('RELOAD',json.dumps(proof),flush=True);assert equal

if __name__=='__main__':
    if sys.argv[1]!='evaluate':T.runtime()
    globals()[sys.argv[1]]()
