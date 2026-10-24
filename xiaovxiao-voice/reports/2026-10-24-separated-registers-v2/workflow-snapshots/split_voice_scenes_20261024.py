"""Independent formal/daily corpora; preserve colloquial wording, never rewrite ASR."""
from pathlib import Path
import os,sys,json,re,time,collections,csv
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['ATEN_CPU_CAPABILITY']='default'
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import prepare_qwen_voice_20261024 as P
C=P.C;ROOT=P.ROOT;PREV=P.PREV;BASE=ROOT/'voices/human-a-v1/local/scene-split-20261024'
BASE.mkdir(exist_ok=True);read,write,rows,append,sha,norm=P.read,P.write,P.rows,P.append,P.sha,P.norm
TERMS=r'啥子|咋个|啷个|晓得|要得|不得|莫得|没得|噻|嘞|家头|安逸|巴适|得行|遭不住|招不住|爪子|勒个'
PROMPT='口语转写，保留重复、语气词和实际用词，不改写为书面普通话。录音中可能出现：啥子、咋个、晓得、要得、没得、不得行、噻、嘞、家头、安逸、耍。只转写实际听到的内容。'

def ok(q):return q and q['SIG']>=3.2 and q['BAK']>=3.3 and q['OVRL']>=2.95
def write_audio(p,a,sr):
    a=resample_poly(a,24000,sr).astype(np.float32)if sr!=24000 else a
    gain=min(10**((-23-20*np.log10(np.sqrt(np.mean(a*a))+1e-12))/20),.85/max(abs(a).max(),1e-8));a=a*gain;sf.write(p,a,24000,subtype='PCM_24');return len(a)/24000

def prepare():
    manifest=BASE/'candidates.jsonl'
    if manifest.exists():print('CANDIDATES_ALREADY_FROZEN',len(rows(manifest)));return
    ann=read(PREV/'annotations.json')['rows'];enh={r['id']:r for r in rows(PREV/'enhancement.jsonl')};inventory={r['id']:r for r in read(PREV/'inventory.json')['rows']};score=P.quality_model();output=[]
    (BASE/'daily-candidates').mkdir(exist_ok=True)
    for r in ann:
        texts=r['raw_transcripts'].get('all',{});ct=r['clean_transcripts'].get('all',{})
        if not 3<=r['duration']<=20 or (r.get('speaker_centroid_cosine')or 0)<.4 or (r.get('window_min_cosine')or 0)<.65 or r['event_candidates']:continue
        if not any(re.search(TERMS,t)for t in list(texts.values())+list(ct.values())):continue
        if max((r.get('original_quality')or{}).get('OVRL',0),(r.get('clean_quality')or{}).get('OVRL',0))<2.95:continue
        a,sr=sf.read(r['path'],dtype='float32');chosen=None;quality=r['original_quality'];variant='original';diagnostics=[]
        if ok(quality):chosen=a
        elif r['id']in enh:
            b,bs=sf.read(enh[r['id']]['path'],dtype='float32');tail=len(a)-len(b)
            if sr==bs==16000 and 0<=tail<160:
                overlap=len(b);b=np.r_[b,np.zeros(tail,dtype=np.float32)]if tail else b;active=abs(a)>np.percentile(abs(a),65);corr=float(np.corrcoef(a[active],b[active])[0,1]);scale=float(np.dot(a[active],b[active])/max(np.dot(b[active],b[active]),1e-12))
                if corr>=.85:
                    for wet in (.25,.5):
                        x=(1-wet)*a+wet*b*scale
                        if tail:x[overlap:]=a[overlap:]
                        temp=BASE/'daily-candidates'/f"{r['id']}_mix{int(wet*100)}.wav";sf.write(temp,x,16000,subtype='PCM_24');q=score(str(temp),16000,False);q={k:float(q[k])for k in ('SIG','BAK','OVRL','P808_MOS')};diagnostics.append(dict(wet=wet,quality=q,correlation=corr))
                        if ok(q):chosen=x;quality=q;variant=f'light_restoration_{wet}';break
        if chosen is None:continue
        # Preserve native bandwidth for unprocessed originals, where available.
        native=False
        if variant=='original':
            original=inventory[r['source_id']];full,fs=sf.read(original['source'],dtype='float32');full=full.mean(axis=1)if full.ndim>1 else full;crop=full[round(r['source_start']*fs):round(r['source_end']*fs)]
            if abs(len(crop)/fs-r['duration'])<.02:chosen,sr=crop,fs;native=True
        path=BASE/'daily-candidates'/(r['id']+'.wav');duration=write_audio(path,chosen,sr)
        output.append(dict(id=r['id'],scene='daily',source_id=r['source_id'],source_start=r['source_start'],source_end=r['source_end'],day=r['day'],split_group=r['split_group'],split=r['proposed_split'],audio=str(path),sha256=sha(path),duration=duration,variant=variant,quality=quality,raw_transcripts=texts,old_clean_transcripts=ct,dialect_evidence={k:re.findall(TERMS,t)for k,t in texts.items()if re.search(TERMS,t)},processing_diagnostics=diagnostics,native_24k_source=native,emotion=r['emotion'],human_audio_reviewed=False))
    formal=ROOT/'voices/human-a-v1/local/dataset_draft';sources={r['source'].split('\\')[-1]:r for r in read(ROOT/'dub/out/human-reference-2026-10-23/source-analysis.json')['rows']};(BASE/'formal-candidates').mkdir(exist_ok=True)
    for i,r in enumerate(rows(formal/'manifest.draft.jsonl')):
        key='formal_'+Path(r['audio']).stem;src=sources[r['source_name']];assert sha(src['source'])==src['source_sha256'];a,sr=sf.read(src['decoded'],dtype='float32');st,en=r['source_range'];a=a[round(st*sr):round(en*sr)];p=BASE/'formal-candidates'/(key+'.wav');duration=write_audio(p,a,sr)
        analysis=BASE/'formal-candidates'/(key+'_analysis16.wav');sf.write(analysis,resample_poly(a,16000,sr),16000,subtype='PCM_24')
        q=score(str(analysis),16000,False);q={k:float(q[k])for k in ('SIG','BAK','OVRL','P808_MOS')}
        output.append(dict(id=key,scene='formal',source_id='formal_'+src['source_sha256'][:16],source_start=st,source_end=en,source_original_name=r['source_name'],source_original_sha256=src['source_sha256'],split_group=r['audio'],split='val'if i==3 else'test'if i==8 else'train',audio=str(p),sha256=sha(p),duration=duration,variant='original',quality=q,raw_transcripts=dict(para=r['text']),emotion='FORMAL_PRESENTATION',human_audio_reviewed=False))
    manifest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in output),encoding='utf-8');print('PREPARED',collections.Counter(r['scene']for r in output),flush=True)

def asr():
    add_formal_b()
    add_formal_clean_references()
    extend_daily_from_verified_enhancement()
    import torch
    from transformers import AutoModelForMultimodalLM,AutoProcessor
    from opencc import OpenCC
    torch.set_num_threads(1);torch.set_num_interop_threads(1);done={(r['id'],r['variant'])for r in rows(BASE/'retranscription.jsonl')};modelpath=str(ROOT/'models/Qwen3-ASR-1.7B-hf');proc=AutoProcessor.from_pretrained(modelpath,local_files_only=True);model=AutoModelForMultimodalLM.from_pretrained(modelpath,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval();conv=OpenCC('t2s');todo=[r for r in rows(BASE/'candidates.jsonl')if (r['id'],'chosen')not in done];t0=time.monotonic()
    for i,r in enumerate(todo):
        a,sr=sf.read(r['audio'],dtype='float32');a=resample_poly(a,16000,sr).astype(np.float32);prompt=PROMPT if r['scene']=='daily'else'正式作品介绍旁白，逐字转写实际内容。'
        inp=proc.apply_transcription_request(audio=[a],language=None,prompt=[prompt],audio_kwargs={'sampling_rate':16000}).to('cuda',dtype=torch.bfloat16)
        with torch.inference_mode():tokens=model.generate(**inp,max_new_tokens=320,do_sample=False)
        text=conv.convert(proc.decode(tokens[:,inp['input_ids'].shape[1]:],return_format='transcription_only')[0]);append(BASE/'retranscription.jsonl',dict(id=r['id'],variant='chosen',text=text,norm=norm(text),prompted=True,prompt_is_not_independent_ground_truth=True))
        if (i+1)%10==0 or i==len(todo)-1:print('RETRANSCRIBE',i+1,len(todo),'eta_s',round((time.monotonic()-t0)/(i+1)*(len(todo)-i-1),1),flush=True)

def add_formal_b():
    manifest=BASE/'candidates.jsonl'
    if any(r['id']=='formal_1-6_referenceB'for r in rows(manifest)):return
    source=next(r for r in read(ROOT/'dub/out/human-reference-2026-10-23/source-analysis.json')['rows']if Path(r['source']).name=='1-6.m4a')
    reference=next(r for r in read(ROOT/'dub/out/human-reference-2026-10-23/references.json')['rows']if r['id']=='B')
    a,sr=sf.read(source['decoded'],dtype='float32');a=a[round(reference['start']*sr):round(reference['end']*sr)];p=BASE/'formal-candidates/formal_1-6_referenceB.wav';duration=write_audio(p,a,sr);ap=p.with_name(p.stem+'_analysis16.wav');sf.write(ap,resample_poly(a,16000,sr),16000,subtype='PCM_24');q=P.quality_model()(str(ap),16000,False)
    append(manifest,dict(id='formal_1-6_referenceB',scene='formal',source_id='formal_'+source['source_sha256'][:16],source_start=reference['start'],source_end=reference['end'],source_original_name='1-6.m4a',source_original_sha256=source['source_sha256'],split_group='1-6_referenceB',split='train',audio=str(p),sha256=sha(p),duration=duration,variant='original',quality={k:float(q[k])for k in ('SIG','BAK','OVRL','P808_MOS')},raw_transcripts={},provided_reference_text=reference['text'],emotion='FORMAL_PRESENTATION',human_audio_reviewed=False))

def extend_daily_from_verified_enhancement():
    manifest=BASE/'candidates.jsonl';done={r['id']for r in rows(manifest)};enh={r['id']:r for r in rows(PREV/'enhancement.jsonl')}
    for r in read(PREV/'annotations.json')['rows']:
        if r['id']in done or r['id']not in enh or not 3<=r['duration']<=20:continue
        if (r.get('speaker_centroid_cosine')or 0)<.4 or (r.get('window_min_cosine')or 0)<.65 or r['event_candidates']:continue
        text=r['raw_transcripts'].get('all',{});clean=r['clean_transcripts'].get('all',{});pros=r.get('prosody',{})
        if not any(re.search(TERMS,t)for t in list(text.values())+list(clean.values())):continue
        if not ok(r['clean_quality'])or (r.get('clean_identity_cosine')or 0)<.9:continue
        if not pros.get('prosody_preservation_pass')or pros['f0_median_abs_delta_semitones']>.25 or pros['f0_p90_abs_delta_semitones']>1 or pros['voiced_overlap']<.65:continue
        a,sr=sf.read(enh[r['id']]['path'],dtype='float32');p=BASE/'daily-candidates'/(r['id']+'.wav');duration=write_audio(p,a,sr)
        append(manifest,dict(id=r['id'],scene='daily',source_id=r['source_id'],source_start=r['source_start'],source_end=r['source_end'],day=r['day'],split_group=r['split_group'],split=r['proposed_split'],audio=str(p),sha256=sha(p),duration=duration,variant='enhanced_strict_pitch_and_identity_preservation',quality=r['clean_quality'],raw_transcripts=text,old_clean_transcripts=clean,dialect_evidence={k:re.findall(TERMS,t)for k,t in text.items()if re.search(TERMS,t)},preservation=dict(cosine=r['clean_identity_cosine'],prosody=pros),native_24k_source=False,emotion=r['emotion'],human_audio_reviewed=False))

def add_formal_clean_references():
    manifest=BASE/'candidates.jsonl';done={r['id']for r in rows(manifest)};sources=read(ROOT/'dub/out/human-reference-2026-10-23/source-analysis.json')['rows'];score=P.quality_model()
    for ref in read(ROOT/'dub/out/human-reference-2026-10-23/references.json')['rows']:
        key='formal_clean_reference_'+ref['id']
        if key in done:continue
        src=next(r for r in sources if Path(r['source']).stem==ref['source_file']);assert sha(ref['path'])==ref['sha256'];a,sr=sf.read(ref['path'],dtype='float32');p=BASE/'formal-candidates'/(key+'.wav');duration=write_audio(p,a,sr);q=score(ref['path'],16000,False)
        append(manifest,dict(id=key,scene='formal',source_id='formal_'+src['source_sha256'][:16],source_start=ref['start'],source_end=ref['end'],source_original_name=ref['source_file']+'.m4a',source_original_sha256=src['source_sha256'],split_group=key,split='train',audio=str(p),sha256=sha(p),duration=duration,variant='existing_real_recording_ZipEnhancer_reference',quality={k:float(q[k])for k in ('SIG','BAK','OVRL','P808_MOS')},raw_transcripts={},provided_reference_text=ref['text'],emotion='FORMAL_PRESENTATION',human_audio_reviewed=False))

def freeze():
    from difflib import SequenceMatcher
    trans={r['id']:r for r in rows(BASE/'retranscription.jsonl')};sv={r['id']:r for r in rows(BASE/'sense.jsonl')};nano={r['id']:r for r in rows(BASE/'nano.jsonl')};output=[];review=[]
    for r in rows(BASE/'candidates.jsonl'):
        t=trans.get(r['id'],{}).get('text','');alltext=dict(r['raw_transcripts']);alltext['sense_fresh']=sv.get(r['id'],{}).get('text','');independent={k:v for k,v in alltext.items()if k not in ('qwen',)and norm(v)}
        if r['scene']=='daily'and r['id']in nano:alltext['nano']=nano[r['id']]['text'];independent['nano']=nano[r['id']]['text']
        # Distinguish homographic de/ta spelling from actual lexical/dialect edits.
        def spelling(x):
            text=norm(x).translate(str.maketrans({'得':'的','地':'的','她':'他','它':'他'}))
            return text.replace('小念','晓念').replace('代办','待办')if r['scene']=='formal'else text
        support=[k for k,v in independent.items()if spelling(v)==spelling(t)]
        reason=[];chosen=t
        # Independent raw recognizers are kept as separate evidence sources.
        # Prefer an agreed verbatim dialect reading over an ASR paraphrase.
        groups={}
        for k,v in alltext.items():
            if not norm(v):continue
            key=spelling(v);g=groups.setdefault(key,dict(models=set(),text=v));g['models'].add('sense'if k=='sense_fresh'else k)
            if k in ('qwen','sense_fresh'):g['text']=v
        agreed=[g for g in groups.values()if len(g['models'])>=2]
        dialect_agreed=[g for g in agreed if re.search(TERMS,norm(g['text']))]
        if r['scene']=='daily'and dialect_agreed:
            g=max(dialect_agreed,key=lambda g:(len(g['models']),len(norm(g['text']))));chosen=g['text'];support=sorted(g['models'])
        elif not support:
            if agreed:
                g=max(agreed,key=lambda g:len(g['models']));chosen=g['text'];support=sorted(g['models'])
            else:reason.append('content_disagreement_needs_listening_not_normalized')
        words=re.findall(TERMS,norm(chosen))
        # Register belongs to the actual chat recording, not to whether the
        # recognizer preserved a written dialect particle in its transcript.
        if not 5<=len(norm(chosen))<=160:reason.append('text_length')
        if not .7<=len(norm(chosen))/r['duration']<=8:reason.append('speech_rate')
        if r.get('evidence_scope')=='contiguous_parent_ASR_agreement'and r['id']in sv:
            if SequenceMatcher(None,spelling(chosen),spelling(sv[r['id']]['text']),autojunk=False).ratio()<.7:reason.append('fresh_crop_transcription_gross_disagreement')
        if r['scene']=='formal'and not (r['quality']['SIG']>=3 and r['quality']['OVRL']>=2.9):reason.append('formal_audio_quality_review')
        chosen=C.plain(chosen)
        if r['scene']=='formal':chosen=chosen.replace('小念','晓念').replace('代办','待办')
        item=dict(r,text=chosen,asr_context_transcript=t,independent_support=support,verbatim_dialect_words=words,dialect_core=bool(words),repeats_and_self_repairs_preserved=True,source_register=r['scene'],human_audio_reviewed=False,accepted=not reason,exclusion_reasons=reason)
        review.append(item)
        if not reason:output.append(item)
    keep=[]
    # Prefer the intact B formal passage, and the largest correctly transcribed
    # dialect passage; never keep overlapping fragments of the same source.
    for r in sorted(output,key=lambda r:(0 if r['id'].startswith('formal_clean_reference_')else 1,-r['duration'])):
        overlap=any(q['source_id']==r['source_id']and min(q['source_end'],r['source_end'])-max(q['source_start'],r['source_start'])>.05 for q in keep)
        duplicate=any(q['scene']==r['scene']and SequenceMatcher(None,norm(q['text']),norm(r['text']),autojunk=False).ratio()>=.9 for q in keep)
        if overlap or duplicate:r['accepted']=False;r['exclusion_reasons'].append('overlap_or_duplicate')
        else:keep.append(r)
    assert {r['source_id']for r in keep if r['scene']=='formal'}.isdisjoint({r['source_id']for r in keep if r['scene']=='daily'})
    summaries={}
    for scene in ('formal','daily'):
        dest=BASE/scene;dest.mkdir(exist_ok=True);selected=[r for r in keep if r['scene']==scene]
        if (dest/'training-result.json').exists():
            assert read(dest/'training-result.json')['snapshot_sha256']==sha(dest/'snapshot.json')
            frozen={r['id']:dict(r,split=s)for s in ('train','val','test')for r in rows(dest/f'{s}_raw.jsonl')}
            for entry in review:
                if entry['scene']==scene and entry['id']in frozen:entry['split']=frozen[entry['id']]['split'];entry['text']=frozen[entry['id']]['text']
            summaries[scene]=read(dest/'selection-summary.json');print('KEEP_TRAINED_SCENE_IMMUTABLE',scene,flush=True);continue
        if scene=='formal':
            # Separate natural phrases; only two original recordings exist, so
            # these holdouts are not independent recording-session evidence.
            for r in selected:r['split']='val'if r['id']=='formal_1-1_01'else'test'if r['id']=='formal_1-1_07'else'train'
            refsrc=ROOT/'voices/human-a-v1/local/reference.wav';reftext=read(ROOT/'voices/human-a-v1/profile.json')['reference_text'];refid='original_1-1_formal_A';language='Chinese'
        else:
            reference=next((r for r in selected if r['id']=='dfdd20343535d6fe_000'),None);assert reference is not None,'Daily dialect reference must be independently transcribed and admitted'
            refsrc=Path(reference['audio']);reftext=reference['text'];refid=reference['id'];language='Auto'
            if not any(r['split']=='val'for r in selected):
                groups=collections.defaultdict(list)
                for r in selected:
                    if r['split']=='train'and r['split_group']!=reference['split_group']:groups[r['split_group']].append(r)
                assert groups,'No separate daily recording group for validation'
                reserve=max(groups,key=lambda g:(min(len(groups[g]),2),sum(r['duration']for r in groups[g])))
                for r in selected:
                    if r['split_group']==reserve:r['split']='val'
        a,sr=sf.read(refsrc,dtype='float32');rp=dest/'reference24.wav';sf.write(rp,resample_poly(a,24000,sr)if sr!=24000 else a,24000,subtype='PCM_24')
        write(dest/'reference.json',dict(id=refid,text=reftext,path=str(rp),sha256=sha(rp),scene=scene,source_is_real_human_recording=True))
        for split in ('train','val','test'):
            items=[dict(id=r['id'],audio=r['audio'],audio_sha256=r['sha256'],text=r['text'],duration=r['duration'],ref_audio=str(rp),language=language,emotion=r['emotion'],source_id=r['source_id'],split_group=r['split_group'],dialect_words=r['verbatim_dialect_words'],dialect_core=r['dialect_core'])for r in selected if r['split']==split]
            (dest/f'{split}_raw.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n'for x in items),encoding='utf-8')
        summary=dict(scene=scene,count=len(selected),minutes=sum(r['duration']for r in selected)/60,splits={s:dict(count=sum(r['split']==s for r in selected),minutes=sum(r['duration']for r in selected if r['split']==s)/60)for s in ('train','val','test')},sources=len({r['source_id']for r in selected}),word_counts=dict(collections.Counter(w for r in selected for w in r['verbatim_dialect_words'])),human_audio_reviewed_count=0,forced_mandarin_rewriting=False,session_independent_holdout=scene=='daily',source_files_disjoint_from_other_scene=True)
        if scene=='daily':
            assert all({r['split_group']for r in selected if r['split']==s}.isdisjoint({r['split_group']for r in selected if r['split']==t})for s,t in [('train','val'),('train','test'),('val','test')])
        print('SCENE_COUNTS',scene,summary['splits'],flush=True)
        write(BASE/'pending-annotations.json',dict(rows=review))
        assert summary['splits']['train']['count']>=3 and summary['splits']['val']['count']>=1 and summary['splits']['test']['count']>=1
        write(dest/'selection-summary.json',summary);write(dest/'snapshot.json',dict(scene=scene,summary=summary,manifests={s:sha(dest/f'{s}_raw.jsonl')for s in ('train','val','test')},audio_hashes={r['id']:r['sha256']for r in selected},reference_sha256=sha(rp)))
        summaries[scene]=summary
    write(BASE/'annotations.json',dict(rows=review));write(BASE/'scene-summary.json',summaries)
    with (BASE/'annotations.tsv').open('w',encoding='utf-8-sig',newline='')as f:
        keys=['scene','id','source_id','accepted','split','duration','text','verbatim_dialect_words','independent_support','exclusion_reasons','variant','human_audio_reviewed'];w=csv.DictWriter(f,fieldnames=keys,delimiter='\t',extrasaction='ignore');w.writeheader();w.writerows(review)
    print(json.dumps(summaries,ensure_ascii=False,indent=2),flush=True)

def sense():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);model=C.model(ROOT/'models/SenseVoiceSmall');done={r['id']for r in rows(BASE/'sense.jsonl')};todo=[r for r in rows(BASE/'candidates.jsonl')if r['id']not in done]
    for i,r in enumerate(todo):
        a,sr=sf.read(r['audio'],dtype='float32');a=resample_poly(a,16000,sr).astype(np.float32);t=model.generate(input=a,fs=16000,language='auto',use_itn=False,disable_pbar=True)[0]['text'];append(BASE/'sense.jsonl',dict(id=r['id'],text=t,norm=norm(t)))
        if (i+1)%10==0 or i==len(todo)-1:print('SENSE',i+1,len(todo),flush=True)

def nano():
    import torch
    from funasr import AutoModel
    torch.set_num_threads(1);torch.set_num_interop_threads(1);sys.path.insert(0,'E:/Fun-ASR-source')
    path=ROOT/'models/Fun-ASR-Nano-2512';model=AutoModel(model=str(path),trust_remote_code=True,remote_code='E:/Fun-ASR-source/model.py',device='cuda:0',ncpu=1,disable_update=True,disable_pbar=True,llm_conf=dict(init_param_path=str(path/'Qwen3-0.6B'),llm_dtype='bf16',load_kwargs=dict(attn_implementation='sdpa')),tokenizer_conf=dict(init_param_path=str(path/'Qwen3-0.6B')))
    done={r['id']for r in rows(BASE/'nano.jsonl')};todo=[r for r in rows(BASE/'candidates.jsonl')if r['scene']=='daily'and r['id']not in done];t0=time.monotonic()
    for i,r in enumerate(todo):
        a,sr=sf.read(r['audio'],dtype='float32');a=resample_poly(a,16000,sr).astype(np.float32);result=model.generate(input=[torch.from_numpy(a)],cache={},batch_size=1,language='中文',itn=False,hotwords=[],max_new_tokens=256,disable_pbar=True)[0]
        append(BASE/'nano.jsonl',dict(id=r['id'],text=result.get('text',''),norm=norm(result.get('text','')),ctc_text=result.get('ctc_text'),timestamps=result.get('timestamps',[]),independent_model='Fun-ASR-Nano-2512',itn=False))
        if (i+1)%10==0 or i==len(todo)-1:print('NANO',i+1,len(todo),'eta',round((time.monotonic()-t0)/(i+1)*(len(todo)-i-1),1),flush=True)

def recover_phrases():
    from difflib import SequenceMatcher
    manifest=BASE/'candidates.jsonl';candidates=rows(manifest);known={r['id']for r in candidates};accepted=[r for r in read(BASE/'annotations.json')['rows']if r['accepted']and r['scene']=='daily'];accepted_ids={r['id']for r in accepted};nano={r['id']:r for r in rows(BASE/'nano.jsonl')};qw={r['id']:r for r in rows(BASE/'retranscription.jsonl')};sense={r['id']:r for r in rows(BASE/'sense.jsonl')};score=P.quality_model();added=[]
    for r in candidates:
        if r['scene']!='daily'or r['id']in accepted_ids or r['id']not in nano:continue
        n=nano[r['id']];ts=[z for z in n['timestamps']if norm(z['token'])];text=''.join(norm(z['token'])for z in ts)
        if text!=norm(n['text']):continue
        offsets=[0]
        for z in ts:offsets.append(offsets[-1]+len(norm(z['token'])))
        a,sr=sf.read(r['audio'],dtype='float32');a16=resample_poly(a,16000,sr).astype(np.float32);block=160;energy=np.mean(a16[:len(a16)//block*block].reshape(-1,block)**2,axis=1);p90=float(np.percentile(energy,90));threshold=min(p90*.1,max(p90*10**(-25/10),float(np.percentile(energy,10))*2))
        def cut(left,right):
            lo=max(0,int(left*100));hi=min(len(energy),int(right*100));mask=energy[lo:hi]<threshold
            edges=np.diff(np.r_[False,mask,False].astype(int));runs=list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
            valid=[(x,y)for x,y in runs if y-x>=6]
            if not valid:return None
            x,y=max(valid,key=lambda p:p[1]-p[0]);return (lo+(x+y)/2)/100
        others=dict(r['raw_transcripts']);others['qwen_context']=qw.get(r['id'],{}).get('text','');others['sense_fresh']=sense.get(r['id'],{}).get('text','');proposals=[]
        sp=lambda x:norm(x).translate(str.maketrans({'得':'的','地':'的','她':'他','它':'他'}))
        for model,t in others.items():
            if not norm(t):continue
            for match in SequenceMatcher(None,sp(text),sp(t),autojunk=False).get_matching_blocks():
                if match.size<10:continue
                starts=[i for i in range(len(ts))if offsets[i]>=match.a and offsets[i]<match.a+match.size and (i==0 or ts[i]['start_time']-ts[i-1]['end_time']>=.08)]
                ends=[i for i in range(len(ts))if offsets[i+1]<=match.a+match.size and offsets[i+1]>match.a and(i==len(ts)-1 or ts[i+1]['start_time']-ts[i]['end_time']>=.08)]
                for first in starts:
                    st=0. if first==0 else cut(ts[first-1]['end_time'],ts[first]['start_time'])
                    if st is None:continue
                    for last in ends:
                        if last<first or offsets[last+1]-offsets[first]<10:continue
                        en=r['duration']if last==len(ts)-1 else cut(ts[last]['end_time'],ts[last+1]['start_time'])
                        if en is None or not 3<=en-st<=12:continue
                        phrase=text[offsets[first]:offsets[last+1]];other=norm(t)[match.b+offsets[first]-match.a:match.b+offsets[last+1]-match.a]
                        if not re.search(TERMS+r'|然后|我我|哎|嘛|啊|呀|哦',phrase):continue
                        global_st=r['source_start']+st;global_en=r['source_start']+en
                        if any(z['source_id']==r['source_id']and min(z['source_end'],global_en)-max(z['source_start'],global_st)>.05 for z in accepted):continue
                        proposals.append(dict(st=st,en=en,text=phrase,other=other,model=model,has_dialect=bool(re.search(TERMS,phrase))))
        chosen=[]
        for item in sorted(proposals,key=lambda z:(not z['has_dialect'],-(z['en']-z['st']))):
            if all(item['en']<=z['st']or item['st']>=z['en']for z in chosen):chosen.append(item)
            if len(chosen)==2:break
        for j,item in enumerate(chosen):
            key=r['id']+f"_native_{round(item['st']*1000)}_{round(item['en']*1000)}"
            if key in known:continue
            clip=a[round(item['st']*sr):round(item['en']*sr)];p=BASE/'daily-candidates'/(key+'.wav');duration=write_audio(p,clip,sr);ap=p.with_name(p.stem+'_q16.wav');x,fs=sf.read(p,dtype='float32');sf.write(ap,resample_poly(x,16000,fs),16000,subtype='PCM_24');q=score(str(ap),16000,False);quality={k:float(q[k])for k in ('SIG','BAK','OVRL','P808_MOS')}
            if not ok(quality):continue
            name='sense'if item['model']=='sense_fresh'else item['model'];entry=dict(r,id=key,parent_id=r['id'],audio=str(p),sha256=sha(p),duration=duration,source_start=r['source_start']+item['st'],source_end=r['source_start']+item['en'],variant=r['variant']+'_continuous_phrase',quality=quality,raw_transcripts={'nano':item['text'],name:item['other']},evidence_scope='contiguous_parent_ASR_agreement',parent_match_models=['nano',name],internal_audio_edits=0,real_low_energy_boundary_required=True)
            append(manifest,entry);added.append(entry);known.add(key)
    write(BASE/'phrase-recovery-summary.json',dict(added=len(added),minutes=sum(r['duration']for r in added)/60,source='contiguous original-recording spans; no synthesized training audio',proof='native CTC token timing plus exact cross-model phrase agreement and >=60ms measured low-energy boundaries; fresh crop ASR next'))
    print('RECOVERED',len(added),sum(r['duration']for r in added)/60,flush=True)

def supplement_conversation():
    """Actual chat turns with repairs/questions; never infer accent from spelling."""
    manifest=BASE/'candidates.jsonl';done={r['id']for r in rows(manifest)};inventory={r['id']:r for r in read(PREV/'inventory.json')['rows']};added=[]
    for r in read(PREV/'annotations.json')['rows']:
        if not r.get('eligible')or r['id']in done or r['id']=='4d1386f0acb32da1_000_p000':continue
        t=r['text'];plain=norm(t);rep=bool(re.search(r'(.{1,3})\1',plain));particle=bool(re.search('嘛|呀|哎|哦|诶|唉|呢|噻|嘞|哟|啊|嗯',t));question=bool(re.search(r'你.*[？?]|[？?].*你',t));formula=bool(re.search('我感觉|我觉得|我的意思|没有啊|不是.*就是|不对|好像|我才|我刚',t));repair=bool(re.search('然后.*然后|那个.*那个|我我|今天今天|昨天昨天',t))
        if not(repair or formula or(particle and(rep or'你'in t))or question):continue
        if r['event_candidates']or not 3<=r['duration']<=16:continue
        native=False
        if ok(r['original_quality']):
            src=inventory[r['source_id']];a,sr=sf.read(src['source'],dtype='float32');a=a.mean(axis=1)if a.ndim>1 else a;a=a[round(r['source_start']*sr):round(r['source_end']*sr)];variant='original';quality=r['original_quality'];native=sr>=24000;proof={}
        else:
            pros=r.get('prosody',{})
            if not ok(r['clean_quality'])or(r.get('clean_identity_cosine')or 0)<.9 or not pros.get('prosody_preservation_pass')or pros['f0_median_abs_delta_semitones']>.25 or pros['f0_p90_abs_delta_semitones']>1 or pros['voiced_overlap']<.65:continue
            a,sr=sf.read(r['selected_audio'],dtype='float32');variant='enhanced_strict_pitch_and_identity_preservation';quality=r['clean_quality'];proof=dict(cosine=r['clean_identity_cosine'],prosody=pros)
        if abs(len(a)/sr-r['duration'])>.02:continue
        p=BASE/'daily-candidates'/(r['id']+'.wav');duration=write_audio(p,a,sr)
        entry=dict(id=r['id'],scene='daily',source_id=r['source_id'],source_start=r['source_start'],source_end=r['source_end'],day=r['day'],split_group=r['split_group'],split=r['proposed_split'],audio=str(p),sha256=sha(p),duration=duration,variant=variant,quality=quality,raw_transcripts=r['raw_transcripts']['all'],old_clean_transcripts=r['clean_transcripts'].get('all',{}),native_24k_source=native,preservation=proof,emotion=r['emotion'],human_audio_reviewed=False,register_evidence=dict(source='genuine_personal_chat',self_repair=repair,conversational_formula=formula,question=question,particle=particle,repetition=rep),accent_classification='not_inferred_from_orthography')
        append(manifest,entry);added.append(entry)
    write(BASE/'conversation-supplement-summary.json',dict(added=len(added),minutes=sum(r['duration']for r in added)/60,policy='actual chat dialogue and repair context; not a claim of measured dialect pronunciation; dialect core retains sampling priority'))
    print('CONVERSATIONAL_ADDED',len(added),sum(r['duration']for r in added)/60,flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
