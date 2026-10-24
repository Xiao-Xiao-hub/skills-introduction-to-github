"""New immutable Qwen corpus: source-faithful labels, gentler candidate recovery."""
from pathlib import Path
import os,sys,json,re,collections,time,importlib.util
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['ATEN_CPU_CAPABILITY']='default'
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import qq_voice_corpus_20261023 as C
ROOT=C.ROOT;PREV=ROOT/'voices/human-a-v1/local/qq-corpus-20261024-v2'
OUT=ROOT/'voices/human-a-v1/local/qwen-corpus-20261024-v1';OUT.mkdir(exist_ok=True)
read,write,rows,append,sha,norm=C.read,C.write,C.rows,C.append,C.sha,C.norm

def quality_model():
    import onnxruntime as ort
    spec=importlib.util.spec_from_file_location('dnsmos',ROOT/'models/DNSMOS/dnsmos_local.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    obj=m.ComputeScore.__new__(m.ComputeScore);opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    obj.onnx_sess=ort.InferenceSession(str(ROOT/'models/DNSMOS/sig_bak_ovr.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
    obj.p808_onnx_sess=ort.InferenceSession(str(ROOT/'models/DNSMOS/model_v8.onnx'),sess_options=opts,providers=['CPUExecutionProvider']);return obj

def quality_ok(q):return q and q['OVRL']>=3 and q['SIG']>=3 and q['BAK']>=3.5
def vote(d):
    c=collections.Counter(norm(t)for t in d.values()if norm(t))
    if not c:return None
    best,count=c.most_common()[0];names=[k for k,v in d.items()if norm(v)==best]
    if count<3 or not any(k in names for k in ('qwen','whisper')):return None
    return dict(text=next(d[k]for k in ('qwen','whisper','sense','para')if k in names),norm=best,models=names)

def gentle():
    ann=read(PREV/'annotations.json')['rows'];enh={r['id']:r for r in rows(PREV/'enhancement.jsonl')}
    flags={r['id']:r['flags']for r in read(PREV/'text-content-review.json')['rows']}
    done={r['id']for r in rows(OUT/'gentle-recovery.jsonl')};score=quality_model();todo=[]
    for r in ann:
        if r.get('eligible') or r['id']in done or flags.get(r['id']):continue
        if not 3<=r['duration']<=16 or (r.get('speaker_centroid_cosine')or 0)<.4 or (r.get('window_min_cosine')or 0)<.65 or r['event_candidates']:continue
        q=r['original_quality'];cq=r['clean_quality'];v=vote(r['raw_transcripts'].get('all',{}));cv=vote(r['clean_transcripts'].get('all',{}))
        if not q or not cq or not v or not cv or v['norm']!=cv['norm']:continue
        if q['SIG']<3.3 or q['OVRL']<2.6 or q['BAK']<3 or not quality_ok(cq):continue
        if r['id']not in enh or not r['prosody'].get('prosody_preservation_pass') or r['clean_identity_cosine']<.75:continue
        if not 1<=len(v['norm'])/r['duration']<=8:continue
        todo.append((r,v))
    print('GENTLE_CANDIDATES',len(todo),flush=True)
    (OUT/'gentle').mkdir(exist_ok=True);t0=time.monotonic()
    for i,(r,v)in enumerate(todo):
        a,sr=sf.read(r['path'],dtype='float32');b,bs=sf.read(enh[r['id']]['path'],dtype='float32');assert sr==bs==16000
        # ZipEnhancer drops the last incomplete 100-sample block. Restore only
        # this sub-10ms tail from the original, never trim or shift any speech.
        tail=len(a)-len(b)
        if not 0<=tail<160:
            append(OUT/'gentle-recovery.jsonl',dict(id=r['id'],accepted=None,reason='unexpected_length_delta',tail_samples=tail));continue
        overlap=len(b)
        if tail:b=np.r_[b,np.zeros(tail,dtype=np.float32)]
        active=abs(a)>np.percentile(abs(a),65);corr=float(np.corrcoef(a[active],b[active])[0,1]);variants=[]
        if corr>=.8:
            scale=float(np.dot(a[active],b[active])/max(np.dot(b[active],b[active]),1e-12));scale=max(.1,min(scale,10))
            for wet in (.25,.5,.75):
                x=(1-wet)*a+wet*b*scale
                if tail:x[overlap:]=a[overlap:]
                gain=min(10**((-23-20*np.log10(np.sqrt(np.mean(x*x))+1e-12))/20),.85/max(abs(x).max(),1e-8));x*=gain
                p=OUT/'gentle'/f"{r['id']}_wet{int(wet*100)}.wav";sf.write(p,x,sr,subtype='PCM_24');q=score(str(p),sr,False);q={k:float(q[k])for k in ('SIG','BAK','OVRL','P808_MOS')}
                ok=bool(quality_ok(q)and all(q[k]>=r['original_quality'][k]-.05 for k in ('SIG','BAK','OVRL')))
                variants.append(dict(path=str(p),wet=wet,quality=q,quality_pass=ok,sha256=sha(p)))
        accepted=next((x for x in variants if x['quality_pass']),None)
        append(OUT/'gentle-recovery.jsonl',dict(id=r['id'],text=v['text'],normalized_text=v['norm'],original_consensus_models=v['models'],corr=corr,variants=variants,accepted=accepted,fresh_checks_required=bool(accepted),raw_tail_restored_samples=tail))
        print('GENTLE',i+1,len(todo),r['id'],accepted['wet']if accepted else None,round(time.monotonic()-t0,1),flush=True)

def verify(stage):
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    selected=[r for r in rows(OUT/'gentle-recovery.jsonl')if r['accepted']]
    path=OUT/f'gentle-{stage}.jsonl';done={r['id']for r in rows(path)};todo=[r for r in selected if r['id']not in done]
    if not todo:return
    if stage=='qwen':
        from transformers import AutoModelForMultimodalLM,AutoProcessor
        from opencc import OpenCC
        p=str(ROOT/'models/Qwen3-ASR-1.7B-hf');processor=AutoProcessor.from_pretrained(p,local_files_only=True);model=AutoModelForMultimodalLM.from_pretrained(p,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval();conv=OpenCC('t2s')
    elif stage=='sense':model=C.model(ROOT/'models/SenseVoiceSmall')
    elif stage=='speaker':model=C.model(ROOT/'models/CAMPlusPlus',device='cpu')
    else:raise ValueError(stage)
    originals={r['id']:r['embedding']for r in rows(PREV/'speaker.jsonl')if 'embedding'in r}
    for r in todo:
        a,sr=sf.read(r['accepted']['path'],dtype='float32')
        if stage=='qwen':
            inp=processor.apply_transcription_request(audio=[a],language=['zh'],audio_kwargs={'sampling_rate':sr}).to(model.device,model.dtype)
            with torch.inference_mode():out=model.generate(**inp,max_new_tokens=256,do_sample=False)
            t=conv.convert(processor.decode(out[:,inp['input_ids'].shape[1]:],return_format='transcription_only')[0]);result=dict(text=t,norm=norm(t),match=norm(t)==r['normalized_text'])
        elif stage=='sense':
            t=model.generate(input=a,fs=sr,language='zh',use_itn=False,disable_pbar=True)[0]['text'];result=dict(text=t,norm=norm(t),match=norm(t)==r['normalized_text'])
        else:
            import parselmouth
            v=model.generate(input=a,fs=sr,disable_pbar=True)[0]['spk_embedding'].detach().cpu().numpy().ravel();v/=np.linalg.norm(v);raw,_=sf.read(next(x for x in read(PREV/'annotations.json')['rows']if x['id']==r['id'])['path'],dtype='float32')
            def pitch(x):return parselmouth.Sound(x,sampling_frequency=sr).to_pitch_ac(time_step=.02,pitch_floor=70,pitch_ceiling=550).selected_array['frequency']
            f,g=pitch(raw),pitch(a);mask=(f>0)&(g>0);delta=abs(12*np.log2(f[mask]/g[mask]));common=float(mask.mean());med=float(np.median(delta))if len(delta)else 999;p90=float(np.percentile(delta,90))if len(delta)else 999
            cosine=float(v@np.asarray(originals[r['id']]));result=dict(raw_cosine=cosine,pitch_median_delta_st=med,pitch_p90_delta_st=p90,voiced_overlap=common,pass_preservation=cosine>=.75 and med<=.5 and p90<=2 and common>=.4 and len(delta)>=10)
        append(path,dict(id=r['id'],**result));print('VERIFY',stage,r['id'],result.get('match',result.get('pass_preservation')),flush=True)

def freeze():
    from rapidfuzz.distance import Levenshtein
    ann=read(PREV/'annotations.json')['rows'];flags={r['id']:r['flags']for r in read(PREV/'text-content-review.json')['rows']}
    additions={r['id']:r for r in rows(OUT/'gentle-recovery.jsonl')if r['accepted']}
    checks={name:{r['id']:r for r in rows(OUT/f'gentle-{name}.jsonl')}for name in ('qwen','sense','speaker')}
    inventory={r['id']:r for r in read(PREV/'inventory.json')['rows']}
    # New lexical/unfinished flags are based on actual text inspection, not
    # guessed replacement words. Keep these clips outside training.
    extra_flags={'81d0eff2b5b37eef_000_p000':['lexical_ambiguity_needs_listening'],
                 'cd7631414c413951_000_s001_p001':['unfinished_clause'],
                 '4d1386f0acb32da1_000_p000':['rejected_muffled_reference_candidate']}
    selected=[];annotations=[]
    for source in ann:
        r=dict(source);key=r['id'];reason=list(flags.get(key,[]))+extra_flags.get(key,[]);ad=additions.get(key)
        accepted=False;path=None;variant=None;text=r['text'];quality=r['predicted_quality']
        if r.get('eligible'):
            accepted=not reason;path=r['selected_audio'];variant=r['selected_variant']
        elif ad:
            proof=all(checks[s].get(key,{}).get('match')for s in ('qwen','sense'))and checks['speaker'].get(key,{}).get('pass_preservation')
            accepted=proof and not reason;path=ad['accepted']['path'];variant='light_restoration';text=ad['text'];quality=ad['accepted']['quality']
            if not proof:reason.append('fresh_audio_checks_not_passed')
        else:reason.extend(r['exclusion_reasons'])
        r.update(qwen_eligible=bool(accepted),qwen_exclusions=reason,qwen_variant=variant,qwen_text=text,qwen_quality=quality,human_audio_reviewed=False,text_review_status='assistant_text_inspected_existing_flags_preserved')
        if accepted:
            r['qwen_source_audio']=path;r['split']=r.get('split',r['proposed_split']);selected.append(r)
        annotations.append(r)
    # Protect all earlier split groups and reject overlapping source material.
    keep=[]
    for r in sorted(selected,key=lambda r:({'test':0,'val':1,'train':2}[r['split']],-r['qwen_quality']['OVRL'])):
        dup=next((q for q in keep if (q['source_id']==r['source_id'] and min(q['source_end'],r['source_end'])-max(q['source_start'],r['source_start'])>.25)or Levenshtein.normalized_similarity(norm(q['qwen_text']),norm(r['qwen_text']))>=.85),None)
        if dup:r['qwen_eligible']=False;r['qwen_exclusions'].append('overlap_or_near_duplicate')
        else:keep.append(r)
    assert all({r['split_group']for r in keep if r['split']==s}.isdisjoint({r['split_group']for r in keep if r['split']==t})for s,t in [('train','val'),('train','test'),('val','test')])
    outaudio=OUT/'audio24';outaudio.mkdir(exist_ok=True)
    for r in keep:
        p=Path(r['qwen_source_audio']);a,sr=sf.read(p,dtype='float32');native=False
        if r['qwen_variant']=='original':
            original=inventory[r['source_id']]
            assert sha(original['source'])==original['source_sha256']
            try:
                full,nsr=sf.read(original['source'],dtype='float32');full=full.mean(axis=1)if full.ndim>1 else full
                crop=full[round(r['source_start']*nsr):round(r['source_end']*nsr)]
                if nsr>=24000 and abs(len(crop)/nsr-r['duration'])<.02:a,sr=crop,nsr;native=True
            except (RuntimeError,ValueError):pass
        a=resample_poly(a,24000,sr).astype(np.float32)if sr!=24000 else a
        gain=min(10**((-23-20*np.log10(np.sqrt(np.mean(a*a))+1e-12))/20),.85/max(abs(a).max(),1e-8));a*=gain;p=outaudio/(r['id']+'.wav');sf.write(p,a,24000,subtype='PCM_24')
        r.update(qwen_audio=str(p),qwen_audio_sha256=sha(p),qwen_duration=len(a)/24000,native_24k_source_used=native,qwen_processing='constant gain; preserve full utterance interior; 16k processed variants resampled not bandwidth restored')
    # Freeze the precise reference which produced user-preferred sample 03.
    ref=next(r for r in ann if r['id']=='da8438c60ff45710_000');p=Path(ref['prepared_path']);assert sha(p)=='2b0ead609a35637d508b2451167e80804fc93c8c994ebed9d608abc50d90708e'
    a,sr=sf.read(p,dtype='float32');rp=OUT/'reference24.wav';sf.write(rp,resample_poly(a,24000,sr),24000,subtype='PCM_24')
    write(OUT/'reference.json',dict(id=ref['id'],original_reference_sha256=sha(p),reference24_sha256=sha(rp),reference24=str(rp),text=ref['text'],user_preferred_synthetic_sample='03_另一模型_未降噪日常参考.wav',reference_is_human_recording=True))
    for split in ('train','val','test'):
        manifest=[dict(id=r['id'],audio=r['qwen_audio'],audio_sha256=r['qwen_audio_sha256'],text=r['qwen_text'],ref_audio=str(rp),duration=r['qwen_duration'],language='Chinese',emotion=r['emotion'],source_id=r['source_id'],split_group=r['split_group'])for r in sorted(keep,key=lambda r:r['id'])if r['split']==split]
        (OUT/f'{split}_raw.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in manifest),encoding='utf-8')
    summary=dict(candidate_segments=len(ann),selected=len(keep),selected_minutes=sum(r['qwen_duration']for r in keep)/60,splits={s:dict(count=sum(r['split']==s for r in keep),minutes=sum(r['qwen_duration']for r in keep if r['split']==s)/60)for s in ('train','val','test')},newly_recovered=sum(r['qwen_variant']=='light_restoration'for r in keep),variants=dict(collections.Counter(r['qwen_variant']for r in keep)),emotions=dict(collections.Counter(r['emotion']for r in keep)),native_24k_clips=sum(r['native_24k_source_used']for r in keep),human_audio_reviewed_count=0,no_validation_or_test_groups_moved=True,original_training_sets_unchanged=True,retained_fillers_and_internal_pauses=True)
    write(OUT/'annotations.json',dict(rows=annotations));write(OUT/'selection-summary.json',summary)
    write(OUT/'snapshot.json',dict(manifests={s:sha(OUT/f'{s}_raw.jsonl')for s in ('train','val','test')},reference_sha256=sha(rp),audio_hashes={r['id']:r['qwen_audio_sha256']for r in keep},summary=summary))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    if sys.argv[1].startswith('verify-'):verify(sys.argv[1].split('-',1)[1])
    else:globals()[sys.argv[1]]()
