"""Keep the user's third-sample adapter and identity; compare narrow delivery fixes."""
import sys,time,json,hashlib
from pathlib import Path
import numpy as np,soundfile as sf,torch
import scene_voice_runtime_20261024 as R
T=R.T;S=R.S;OUT=S.BASE/'daily-expression-r3';OUT.mkdir(exist_ok=True)
SAME='我刚刚才看到消息，等一下就过来。你先别着急，我把这边弄好就来找你。'
CHAT='要得，我这边快弄好了。你先过去嘛，等下我们一起耍。'
PACK=T.ROOT/'models/Qwen3-TTS-Xuan-daily-v2'
SETTINGS=dict(temperature=.8,top_p=.9,subtalker_temperature=.9,subtalker_top_p=1.,max_new_tokens=200)

def synth():
    T.environment();v=R.create_model(PACK);meta=R.select(v,PACK);reference=T.read(S.BASE/'daily/reference.json');assert T.sha(reference['path'])==reference['sha256']
    speaker=torch.load(PACK/'speaker-embedding.pt',map_location='cuda',weights_only=True)[0];manifest=OUT/'manifest.jsonl';done={r['id']for r in T.rows(manifest)};prompt=None
    for key,method,text,nonstream in [('control_same','custom',SAME,True),('b_same','real_context',SAME,True),('c_same','real_context',SAME,False)]:
        if key in done:continue
        T.seed(201);start=time.monotonic();counter=[0]
        def progress(module,args,result):
            counter[0]+=1
            if counter[0]==1 or counter[0]%16==0:
                state=dict(id=key,frames=counter[0],elapsed_s=round(time.monotonic()-start,1));T.write(OUT/'generation-status.json',state);print('FRAMES',json.dumps(state),flush=True)
        hook=v.model.talker.register_forward_hook(progress)
        try:
            if method=='custom':
                wavs,sr=v.generate_custom_voice(text=text,language='Auto',speaker=meta['speaker'],non_streaming_mode=nonstream,**SETTINGS)
            else:
                v.model.config.tts_model_type='base';v.model.tts_model_type='base'
                if prompt is None:
                    prompt=v.create_voice_clone_prompt(ref_audio=reference['path'],ref_text=reference['text'],x_vector_only_mode=False)
                    # The same frozen human identity vector as the user's third sample.
                    # Only real acoustic/text context is added; no synthetic target audio.
                    prompt[0].ref_spk_embedding=speaker.clone();assert torch.equal(prompt[0].ref_spk_embedding,speaker)
                T.seed(201);wavs,sr=v.generate_voice_clone(text=text,language='Auto',voice_clone_prompt=prompt,non_streaming_mode=nonstream,**SETTINGS)
        finally:hook.remove()
        if counter[0]>=SETTINGS['max_new_tokens']:raise RuntimeError('Generation hit bounded codec cap; do not deliver truncated or looping audio')
        p=OUT/(key+'.wav');sf.write(p,wavs[0],sr,subtype='PCM_24');entry=dict(id=key,scene='daily',kind='same'if key.endswith('same')else'chat',mode=method,text=text,path=str(p),sr=sr,duration=len(wavs[0])/sr,seconds=time.monotonic()-start,sha256=T.sha(p),adapter_sha256=meta['adapter_sha256'],speaker_embedding_sha256=meta['speaker_embedding_sha256'],reference_audio_sha256=reference['sha256']if method=='real_context'else None,identity_vector_same_as_user_third=True,reference_is_real_human=method=='real_context',synthetic_reference_used=False,new_training_performed=False,seed=201,settings=dict(SETTINGS,non_streaming_mode=nonstream),codec_frames=counter[0],human_accepted=False)
        T.append(manifest,entry);print('GENERATED',key,round(entry['duration'],2),round(entry['seconds'],1),flush=True)
        if key=='control_same':
            proof=dict(waveform_bit_identical=T.sha(p)==T.sha(S.BASE/'audition/daily_same.wav'),baseline_sha256=T.sha(S.BASE/'audition/daily_same.wav'),new_sha256=T.sha(p));T.write(OUT/'third-reproduction.json',proof);assert proof['waveform_bit_identical']

def evaluate():
    import qq_lora_audition_20261023 as A
    A.OUT=S.PREV;A.SAMPLES=OUT;A.evaluate()

def compare():
    from scipy.signal import resample_poly
    import qq_voice_corpus_20261023 as C
    m=C.model(T.ROOT/'models/CAMPlusPlus',device='cpu')
    def emb(path):
        a,sr=sf.read(path,dtype='float32');a=resample_poly(a,16000,sr).astype(np.float32);v=m.generate(input=a,fs=16000,disable_pbar=True)[0]['spk_embedding'].detach().cpu().numpy().ravel();return v/np.linalg.norm(v)
    third=S.BASE/'audition/daily_same.wav';baseline=emb(third);human=emb(S.BASE/'daily/reference24.wav');d=T.read(OUT/'evaluation.json');vectors={}
    for r in d['rows']:
        e=emb(r['path']);vectors[r['id']]=e;r['cosine_to_user_third']=float(e@baseline);r['cosine_to_original_daily_reference']=float(e@human)
        a,sr=sf.read(r['path'],dtype='float32');a=resample_poly(a,16000,sr).astype(np.float32);windows=[]
        for start in range(0,len(a)-48000+1,32000):
            w=m.generate(input=a[start:start+48000],fs=16000,disable_pbar=True)[0]['spk_embedding'].detach().cpu().numpy().ravel();w/=np.linalg.norm(w);windows.append(dict(start_s=start/16000,cosine_to_utterance=float(w@e),cosine_to_third=float(w@baseline)))
        r['voice_windows']=windows;r['min_window_cosine_to_utterance']=min((w['cosine_to_utterance']for w in windows),default=None)
    d['pairs']={name:float(vectors[x]@vectors[y])for name,x,y in [('context_variants','b_same','c_same')]if x in vectors and y in vectors};d['baseline']=dict(third_sha256=T.sha(third),human_reference_cosine=float(baseline@human));d['human_listening_accepted']=False;d['similarities_are_not_identity_or_emotion_acceptance']=True;T.write(OUT/'evaluation.json',d)
    print('COMPARISON',json.dumps([{k:r.get(k)for k in ('id','duration','cosine_to_user_third','cosine_to_original_daily_reference','longest_internal_low_energy_gap_s','text_verified_by_both_asr')}for r in d['rows']]),flush=True)

def publish():
    import shutil,subprocess
    dest=T.ROOT/'reports/2026-10-24-daily-expression-r3';audio=dest/'audition';audio.mkdir(exist_ok=True);data=T.read(OUT/'evaluation.json');public=[]
    for r in data['rows']:
        assert r['text_verified_by_both_asr']and r['clipped_samples']==0
        assert r['adapter_sha256']==T.read(PACK/'voice.json')['adapter_sha256']
        item=dict(r)
        for key in ('path','preview'):
            p=Path(r[key]);shutil.copy2(p,audio/p.name);item[key]='audition/'+p.name
        assert T.sha(dest/item['path'])==r['sha256'];public.append(item)
    out=dict(data,rows=public);T.write(dest/'audition-checks.json',out)
    for name in ('third-reproduction.json','failed-low-temperature.json'):shutil.copy2(OUT/name,dest/name)
    T.write(dest/'listen-map.json',dict(first_adjustment='b_same',second_adjustment='c_same',previous_third_reproduction='control_same',new_candidate_count=2,control_count=1,target='亲近、轻松带笑，有一点活泼',human_listening_accepted=False))
    snap=dest/'workflow-snapshots';snap.mkdir(exist_ok=True);shutil.copy2(Path(__file__),snap/Path(__file__).name);shutil.copy2(Path('E:/Qwen3-TTS-training/LICENSE'),snap/'UPSTREAM-LICENSE')
    protected=['specs.py','common.py','step1_audition.py','dub.py','dub/xiaonian_v10.json'];assert not subprocess.check_output(['git','diff','HEAD','--',*protected],cwd=T.ROOT,text=True)
    T.write(dest/'completion.json',dict(completed=True,new_training_performed=False,personal_adapter_unchanged=True,speaker_embedding_unchanged=True,synthetic_reference_or_training_audio_used=False,original_toolkit_scripts_unchanged=True,private_reference_uploaded=False,new_candidates=2,human_listening_accepted=False,helper_sha256=T.sha(snap/Path(__file__).name)))
    print('PUBLISHED_TWO_CANDIDATES_PLUS_CONTROL',flush=True)

if __name__=='__main__':
    if sys.argv[1]!='evaluate':T.runtime()
    globals()[sys.argv[1]]()
