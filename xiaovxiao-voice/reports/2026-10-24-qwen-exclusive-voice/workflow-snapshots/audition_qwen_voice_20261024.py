"""New-text Qwen before/after comparison and independent acoustic/text checks."""
import sys,os,time,json
from pathlib import Path
import numpy as np
import soundfile as sf
import train_qwen_voice_20261024 as T
D=T.D;ROOT=T.ROOT;OUT=T.OUT;AUDIO=OUT/'audition';AUDIO.mkdir(exist_ok=True)
TEXTS={
 'warm':'不用急，我们先试一下。要是哪里不顺手，就停下来，一起换个办法。',
 'daily':'你等我一下，我再找找。哦，在这里！刚才还以为弄丢了呢。',
 'happy':'真的呀？那太好了！我还以为今天来不及了，没想到这么快就弄好了。',
 'thoughtful':'我刚开始也有点拿不准。后来试了几次，才发现，慢一点反而更容易想清楚。',
 'continuity':'其实，我想把这件事做得简单一点。想到什么，就先记下来，也不用马上整理得很漂亮。等过一阵子再回头看，有些问题，好像自己就慢慢想明白了。',
}

def synth():
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    T.environment();result=T.read(OUT/'training-result.json');best=Path(result['selected_checkpoint']);assert T.sha(best/'adapter_model.safetensors')==result['checkpoint_sha256']
    v,config=T.load_training();model=v.model;set_peft_model_state_dict(model,load_file(str(best/'adapter_model.safetensors'),device='cuda'))
    model.talker.gradient_checkpointing_disable();model.talker.config.use_cache=True;model.talker.code_predictor.config.use_cache=True;model.eval()
    original=T.read(D.PREV/'annotations.json')['rows'];ref=next(r for r in original if r['id']=='da8438c60ff45710_000');assert T.sha(ref['prepared_path'])==T.read(OUT/'reference.json')['original_reference_sha256']
    prompt=v.create_voice_clone_prompt(ref_audio=ref['prepared_path'],ref_text=ref['text']);manifest=AUDIO/'manifest.jsonl';done={r['id']for r in T.rows(manifest)}
    from difflib import SequenceMatcher
    traintexts=[D.norm(r['text'])for r in T.rows(OUT/'train_raw.jsonl')]
    separation={k:max(SequenceMatcher(None,D.norm(t),q,autojunk=False).ratio()for q in traintexts)for k,t in TEXTS.items()};assert max(separation.values())<.85;T.write(AUDIO/'novel-text-check.json',separation)
    from contextlib import nullcontext
    for mode in ('base','trained'):
        if mode=='trained':
            emb=torch.load(OUT/'speaker-embedding.pt',map_location='cuda',weights_only=True)
            with torch.no_grad():model.talker.model.codec_embedding.weight[3000]=emb[0].to(model.dtype)
            model.config.tts_model_type='custom_voice';model.tts_model_type='custom_voice';model.config.talker_config.spk_id={'xuan':3000};model.config.talker_config.spk_is_dialect={'xuan':False};model.supported_speakers=['xuan']
        with T._disabled(model)if mode=='base'else nullcontext():
            for kind,text in TEXTS.items():
                key=f'{mode}_{kind}'
                if key in done:continue
                T.seed(201);start=time.monotonic()
                if mode=='base':wavs,sr=v.generate_voice_clone(text=text,language='Chinese',voice_clone_prompt=prompt,temperature=.8,top_p=.9,max_new_tokens=1024)
                else:wavs,sr=v.generate_custom_voice(text=text,language='Chinese',speaker='xuan',temperature=.8,top_p=.9,max_new_tokens=1024)
                a=wavs[0];p=AUDIO/(key+'.wav');sf.write(p,a,sr,subtype='PCM_24')
                T.append(manifest,dict(id=key,mode=mode,kind=kind,text=text,path=str(p),sr=sr,duration=len(a)/sr,seconds=time.monotonic()-start,sha256=T.sha(p),checkpoint_sha256=result['checkpoint_sha256']if mode=='trained'else None,reference_sha256=T.sha(ref['prepared_path']),reference_audio_required_at_inference=mode=='base',generation_method='voice_clone'if mode=='base'else'custom_voice_xuan',seed=201,temperature=.8,top_p=.9,human_accepted=False))
                if key=='base_warm':
                    old=D.PREV/'clarity-diagnosis/qwen_new_raw_clone.wav';x,xsr=sf.read(old,dtype='float32');y,ysr=sf.read(p,dtype='float32');n=min(len(x),len(y))
                    T.write(AUDIO/'baseline-reproduction.json',dict(original_sha256=T.sha(old),new_sha256=T.sha(p),same_length=len(x)==len(y),same_sample_rate=xsr==ysr,overlap_max_abs_difference=float(abs(x[:n]-y[:n]).max()),overlap_correlation=float(np.corrcoef(x[:n],y[:n])[0,1]),same_reference_and_text_and_seed=True))
                print('SYNTH',key,round(time.monotonic()-start,1),len(a)/sr,flush=True)
    # Additional diagnostic: learned weights in the original ICL cloning mode.
    key='trained_clone_warm'
    if key not in done:
        model.config.tts_model_type='base';model.tts_model_type='base';T.seed(201);start=time.monotonic()
        wavs,sr=v.generate_voice_clone(text=TEXTS['warm'],language='Chinese',voice_clone_prompt=prompt,temperature=.8,top_p=.9,max_new_tokens=1024);p=AUDIO/(key+'.wav');sf.write(p,wavs[0],sr,subtype='PCM_24')
        T.append(manifest,dict(id=key,mode='trained_clone',kind='warm',text=TEXTS['warm'],path=str(p),sr=sr,duration=len(wavs[0])/sr,seconds=time.monotonic()-start,sha256=T.sha(p),checkpoint_sha256=result['checkpoint_sha256'],reference_audio_required_at_inference=True,seed=201,temperature=.8,top_p=.9,human_accepted=False))
        print('SYNTH',key,time.monotonic()-start,flush=True)

def evaluate():
    # Run from Vox environment, after the Qwen model process has exited.
    # Avoid importing T a second time with conflicting Torch thread setup.
    import qq_lora_audition_20261023 as A
    A.OUT=D.PREV;A.SAMPLES=AUDIO;A.evaluate()

def exported_more():
    from qwen_tts import Qwen3TTSModel
    import torch
    v=Qwen3TTSModel.from_pretrained(str(ROOT/'models/Qwen3-TTS-Xuan-20261024-v1'),device_map='cuda:0',dtype=torch.bfloat16,attn_implementation='sdpa')
    manifest=AUDIO/'manifest.jsonl';done={r['id']for r in T.rows(manifest)};result=T.read(OUT/'training-result.json')
    for kind in ('daily','happy'):
        key='exported_custom_'+kind
        if key in done:continue
        T.seed(201);start=time.monotonic();wavs,sr=v.generate_custom_voice(text=TEXTS[kind],language='Chinese',speaker='xuan',temperature=.8,top_p=.9,max_new_tokens=1024);p=AUDIO/(key+'.wav');sf.write(p,wavs[0],sr,subtype='PCM_24')
        T.append(manifest,dict(id=key,mode='export_smoke',kind=kind,text=TEXTS[kind],path=str(p),sr=sr,duration=len(wavs[0])/sr,seconds=time.monotonic()-start,sha256=T.sha(p),checkpoint_sha256=result['checkpoint_sha256'],reference_audio_required_at_inference=False,generation_method='merged_custom_voice_xuan',seed=201,temperature=.8,top_p=.9,human_accepted=False));print('EXPORTED_SYNTH',key,round(time.monotonic()-start,1),flush=True)

def qwen_check():
    import torch
    from transformers import AutoModelForMultimodalLM,AutoProcessor
    from scipy.signal import resample_poly
    from opencc import OpenCC
    from rapidfuzz.distance import Levenshtein
    p=str(ROOT/'models/Qwen3-ASR-1.7B-hf');proc=AutoProcessor.from_pretrained(p,local_files_only=True);model=AutoModelForMultimodalLM.from_pretrained(p,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval();conv=OpenCC('t2s');data=T.read(AUDIO/'evaluation.json')
    for r in data['rows']:
        if 'qwen'not in r:
            a,sr=sf.read(r['path'],dtype='float32');a=resample_poly(a,16000,sr).astype(np.float32);inp=proc.apply_transcription_request(audio=[a],language=['zh'],audio_kwargs={'sampling_rate':16000}).to('cuda',dtype=torch.bfloat16)
            with torch.inference_mode():out=model.generate(**inp,max_new_tokens=256,do_sample=False)
            t=conv.convert(proc.decode(out[:,inp['input_ids'].shape[1]:],return_format='transcription_only')[0]);r['qwen']=dict(raw=t,match=D.norm(t)==D.norm(r['text']),cer=Levenshtein.distance(D.norm(t),D.norm(r['text']))/len(D.norm(r['text'])))
        r['exact_asr_matches']=sum(r[n]['match']for n in ('para','sense','qwen'));r['text_verified_by_two_of_three_asr']=r['exact_asr_matches']>=2;T.write(AUDIO/'evaluation.json',data);print('ASR',r['id'],r['exact_asr_matches'],flush=True)

def consistency():
    from scipy.signal import resample_poly
    model=D.C.model(ROOT/'models/CAMPlusPlus',device='cpu');data=T.read(AUDIO/'evaluation.json')
    def emb(a):
        v=model.generate(input=a,fs=16000,disable_pbar=True)[0]['spk_embedding'].detach().cpu().numpy().ravel();return v/np.linalg.norm(v)
    for r in data['rows']:
        if r['kind']!='continuity':continue
        a,sr=sf.read(r['path'],dtype='float32');a=resample_poly(a,16000,sr).astype(np.float32);full=emb(a);windows=[]
        for start in range(0,len(a)-48000+1,32000):windows.append(dict(start_s=start/16000,cosine_to_utterance=float(emb(a[start:start+48000])@full)))
        r['continuity_windows']=windows;r['minimum_window_cosine']=min(x['cosine_to_utterance']for x in windows)
        print('CONTINUITY',r['id'],r['minimum_window_cosine'],flush=True)
    T.write(AUDIO/'evaluation.json',data)

if __name__=='__main__':
    if sys.argv[1]!='evaluate':T.runtime()
    globals()[sys.argv[1]]()
