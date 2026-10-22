"""Scoped listener-requested repair: replace group 23b, remove closing narration."""
from pathlib import Path
import gc, json, re, shutil, sys, tempfile
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from c3_video_align import ROOT, SR, read, write, sha, norm, canonical_units, dub
import c3_video_mix as base

OLD=ROOT/'dub/out/c3-video-r2-2026-10-22'
OUT=ROOT/'dub/out/c3-video-r3-2026-10-22'
FINAL=Path('D:/XiaoNian-Video/review/2026-10-22-v10-c3-r3/晓念_V10_C3_音色衔接修订_无片尾旁白.mp4')
PARA=ROOT/'.cache/modelscope/models/iic--speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/snapshots/master'

def generate():
    from common import VoxCPM2Backend
    a=read(OLD/'aligned/alignment.json')
    prev=next(g for g in a['groups']if g['group']=='23a')
    target=next(g for g in a['groups']if g['group']=='23b')
    ref=Path(prev['source']['path']);assert sha(ref)==prev['source']['sha256']
    text=''.join(l['synth']for l in target['lines'])
    prompt=''.join(l['synth']for l in prev['lines'])
    OUT.mkdir(parents=True,exist_ok=True)
    model=VoxCPM2Backend(str(ROOT/'models/VoxCPM2'));rows=[]
    for seed in (137,139):
        path=OUT/f'g23b_s{seed}.wav'
        assert not path.exists()
        wav,sr,used=model._generate(seed,{'cfg':2.,'steps':20},text=text,prompt_wav_path=str(ref),prompt_text=prompt,reference_wav_path=str(ref))
        sf.write(path,wav,sr,subtype='PCM_24')
        rows.append(dict(path=str(path),sha256=sha(path),seed=used,duration=len(wav)/sr,reference=str(ref),reference_sha256=sha(ref),prompt_text=prompt,text=text))
        write(OUT/'candidates.json',dict(rows=rows,lines=target['lines']))
        print('generated',seed,len(wav)/sr,flush=True)

def recognize():
    from funasr import AutoModel
    data=read(OUT/'candidates.json');model=AutoModel(model=str(PARA),device='cuda:0',ncpu=4,disable_update=True,disable_pbar=True)
    for row in data['rows']:
        wav,sr=sf.read(row['path'],dtype='float32');wav=resample_poly(wav,16000,sr).astype(np.float32)
        result=model.generate(input=wav,fs=16000,batch_size=1,disable_pbar=True)[0]
        row['asr']=result;row['match']=norm(result.get('text',''))==norm(row['text'])
        print('ASR',row['seed'],result.get('text'),row['match'],flush=True)
    del model;gc.collect()
    import torch
    torch.cuda.empty_cache()
    # Compare identity with the two actual adjacent takes, using the existing CPU model.
    model=AutoModel(model=str(ROOT/'models/CAMPlusPlus'),device='cpu',ncpu=4,disable_update=True,disable_pbar=True)
    old=read(OLD/'aligned/alignment.json');vectors={}
    for g in old['groups']:
        if g['group']not in ('23a','23b','24'):continue
        wav,sr=sf.read(g['rendered_path'],dtype='float32');wav=resample_poly(wav,16000,sr).astype(np.float32)
        v=model.generate(input=wav,fs=16000,batch_size=1,disable_pbar=True)[0]['spk_embedding'].detach().cpu().numpy().reshape(-1)
        vectors[g['group']]=v/np.linalg.norm(v)
    data['old_neighbor_cosines']={k:float(vectors['23b']@vectors[k])for k in ('23a','24')}
    for row in data['rows']:
        wav,sr=sf.read(row['path'],dtype='float32');wav=resample_poly(wav,16000,sr).astype(np.float32)
        v=model.generate(input=wav,fs=16000,batch_size=1,disable_pbar=True)[0]['spk_embedding'].detach().cpu().numpy().reshape(-1);v/=np.linalg.norm(v)
        row['neighbor_cosines']={k:float(v@vectors[k])for k in ('23a','24')}
        print('neighbor_identity',row['seed'],row['neighbor_cosines'],flush=True)
    write(OUT/'candidates.json',data)

def fit(row,lines):
    text,ts=canonical_units(row['asr'])
    if text!=''.join(norm(l['synth'])for l in lines):return None
    spans=[];i=0
    for line in lines:
        n=len(norm(line['synth']));spans.append([ts[i][0],ts[i+n-1][1]]);i+=n
    candidates=[]
    for tolerance in (0,.04,.08,.12,.16):
        for rate in np.arange(.85,1.101,.001):
            low=max(l['start']-(tolerance if i else 0)-x/rate for i,(l,(x,y))in enumerate(zip(lines,spans)))
            high=min(l['end']+(tolerance if i<len(lines)-1 else 0)-y/rate for i,(l,(x,y))in enumerate(zip(lines,spans)))
            low=max(low,141.11-spans[0][0]/rate+.06)
            if low>high:continue
            # Avoid altering cadence unnecessarily; do not cut inside a phrase.
            score=abs(rate-1.)*3+(low+spans[0][0]/rate-lines[0]['start'])*.2
            candidates.append(dict(rate=float(rate),offset=float(low),score=float(score),tolerance=tolerance,spans=spans))
        if candidates:return min(candidates,key=lambda x:x['score'])
    return None

def render():
    data=read(OUT/'candidates.json');possible=[]
    for row in data['rows']:
        f=fit(row,data['lines'])
        if f and min(row['neighbor_cosines'].values())>=.85:possible.append((f['score']+1-np.mean(list(row['neighbor_cosines'].values())),row,f))
    assert possible,'No correctly transcribed candidate fits subtitle windows'
    _,selected,f=min(possible,key=lambda v:v[0]);write(OUT/'selection.json',dict(selected=selected,fit=f))
    oldalign=read(OLD/'aligned/alignment.json');voice=base.read_track(OLD/'aligned/人声轨_48k24bit.wav')
    source_before=voice.copy();oldg=next(g for g in oldalign['groups']if g['group']=='23b')
    # Remove the prior group by subtracting exactly its stored PCM24 samples.
    oldpiece,_=sf.read(oldg['rendered_path'],dtype='float32');start=round(oldg['start']*SR)
    voice[start:start+len(oldpiece)]-=oldpiece
    assert np.max(np.abs(voice[round(141.36*SR):round(152.80*SR)]))<1e-6
    voice[round(169.5*SR):]=0
    dest=OUT/'g23b_retimed.wav'
    dub.run_ffmpeg('-i',selected['path'],'-af',f"rubberband=tempo={f['rate']:.6f}:pitch=1:transients=smooth:detector=soft:formant=preserved",'-ar',SR,'-c:a','pcm_f32le',dest)
    piece,_=sf.read(dest,dtype='float32')
    neighbors=[g for g in oldalign['groups']if g['group']in ('23a','24')]
    levels=[dub.speech_rms_db(sf.read(g['rendered_path'],dtype='float32')[0],SR)for g in neighbors]
    gain=float(np.mean(levels)-dub.speech_rms_db(piece,SR));piece*=10**(gain/20);piece=dub.fade(piece,SR,ms=4)
    sf.write(OUT/'替换段_2分21至2分33.wav',piece,SR,subtype='PCM_24')
    start=round(f['offset']*SR);end=start+len(piece)
    existing=voice[start:end];overlap=np.abs(existing)>.001
    assert not np.any(overlap & (np.abs(piece)>.001)),'New group overlaps existing speech'
    voice[start:end]+=piece
    # Ensure only the replaced group and requested ending can differ.
    mask=np.ones(len(voice),dtype=bool);mask[min(start,round(oldg['start']*SR)):max(end,round(oldg['end']*SR))]=False;mask[round(169.5*SR):]=False
    assert np.array_equal(voice[mask],source_before[mask])
    assert np.max(np.abs(voice[round(169.5*SR):]))==0
    track=OUT/'人声轨_48k24bit.wav';sf.write(track,voice,SR,subtype='PCM_24')
    cues=[r for r in oldalign['rows']if r['group']not in ('23b','26-27')]
    repaired=[]
    for line,(x,y)in zip(data['lines'],f['spans']):
        repaired.append(dict(group='23b',text=line['text'],start=f['offset']+x/f['rate'],end=f['offset']+y/f['rate'],window=[line['start'],line['end']]))
    cues+=repaired;cues.sort(key=lambda r:r['start'])
    music=base.read_stem(base.MUSIC,len(voice));sfx=base.read_stem(base.SFX,len(voice));bed=music+sfx
    # Keep the R2 mix balance and original background. No new ducking.
    voicegain=read(OLD/'mix-report.json')['voice_gain_db'];combined=bed+(voice*10**(voicegain/20))[:,None]
    mix=OUT/'mix.wav';assert not FINAL.exists() and not mix.exists()
    with tempfile.TemporaryDirectory(prefix='c3-r3-')as tmp:
        raw=Path(tmp)/'raw.wav';sf.write(raw,combined,SR,subtype='FLOAT');loudness=base.loudnorm_two_pass(raw,mix)
    FINAL.parent.mkdir(parents=True,exist_ok=True)
    dub.run_ffmpeg('-i',base.VIDEO,'-i',mix,'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','256k','-ar',SR,'-ac',2,'-movflags','+faststart','-shortest',FINAL)
    validation=base.validate_final(FINAL,base.video_hash(base.VIDEO));stats=base.loudnorm_stats(FINAL)
    assert float(stats['input_tp'])<=-1.5 and -16.5<=float(stats['input_i'])<=-15.5,stats
    write(OUT/'report.json',dict(video=str(FINAL),video_bytes=FINAL.stat().st_size,video_sha256=sha(FINAL),voice_sha256=sha(track),selected=selected,fit=f,replacement_gain_db=gain,
        repaired_cues=repaired,unchanged_other_voice_samples=True,closing_voice_zero_from=169.5,original_background=True,music_duck_db=0,sfx_duck_db=0,voice_gain_db=voicegain,loudness=stats,validation=validation,human_listening=False))
    print('EXPORTED',FINAL,FINAL.stat().st_size,flush=True)

def check():
    from funasr import AutoModel
    model=AutoModel(model=str(ROOT/'models/SenseVoiceSmall'),device='cuda:0',ncpu=4,disable_update=True,disable_pbar=True,trust_remote_code=False)
    path=OUT/'encoded-replacement.wav'
    dub.run_ffmpeg('-ss',141.11,'-i',FINAL,'-t',12.07,'-vn','-ar',16000,'-ac',1,'-c:a','pcm_s16le',path)
    wav,sr=sf.read(path,dtype='float32');r=model.generate(input=wav,fs=sr,language='zh',use_itn=True,disable_pbar=True)[0]
    raw=r.get('text','');plain=re.sub(r'<\|[^|>]+\|>','',raw).strip();expected=read(OUT/'selection.json')['selected']['text']
    result=dict(raw=raw,plain=plain,expected=expected,match=norm(plain)==norm(expected),scope='Actual final encoded audio 141.11-153.18 seconds')
    write(OUT/'encoded-asr.json',result);print('ENCODED_CHECK',result,flush=True)

if __name__=='__main__':
    {'generate':generate,'recognize':recognize,'render':render,'check':check}[sys.argv[1]]()
