"""Restore uncut phrase groups and mix the original audible soundtrack for R2."""
from pathlib import Path
import json
import shutil
import tempfile
import numpy as np
import soundfile as sf
from c3_video_align import ROOT, SR, read, write, sha, norm, dub
from c3_video_r2_plan import OUT, plan
import c3_video_mix as base

FINAL=Path('D:/XiaoNian-Video/review/2026-10-22-v10-c3-r2/晓念_V10_C3_自然语流_音乐增强版.mp4')

def render_voice():
    groups=plan()
    candidate=next(r for r in read(OUT/'closing/r2-plaintext-results.json')['rows']if r['id']=='r2_plaintext_s109')
    assert norm(candidate['sensevoice']['plain'])==norm(candidate['original_text'])
    # The final scene's actual visual timing comes from film/src/v5/scenes.tsx.
    # Keep this complete fresh take in one piece.  No independently pasted tag.
    closing_lines=dub.load_script(dub.SCRIPT)['groups'][-1]['lines']
    groups.append(dict(group='26-27',source=candidate,lines=closing_lines,
        source_cue_spans=[[.19,1.45],[1.75,3.87],[4.47,4.975]],
        fit=dict(rate=1.,offset=170.58,tolerance=0,first_voice_delay=1.02),
        visual_windows=[[169.50,172.12],[170.667,174.50],[175.05,177.55]],
        visual_note='Untouched natural tempo: definitions finish while readable. Brand starts at first visible title around175.05; no inserted pause.'))
    target=OUT/'aligned';(target/'groups').mkdir(parents=True,exist_ok=True)
    track=np.zeros(178*SR,dtype=np.float64);occupancy=np.zeros(len(track),dtype=bool)
    rendered=[];rows=[];overlaps=[]
    with tempfile.TemporaryDirectory(prefix='c3-r2-voice-') as temp:
        for g in groups:
            assert g['fit'] is not None,g['group']
            source=g['source'];path=Path(source['path']);assert sha(path)==source['sha256']
            a,sr=sf.read(path,dtype='float64');assert sr==SR and a.ndim==1
            rate,offset=g['fit']['rate'],g['fit']['offset']
            converted=Path(temp)/f"g{g['group']}.wav"
            filt=f'rubberband=tempo={rate:.6f}:pitch=1:transients=smooth:detector=soft:formant=preserved'
            if rate==1.:
                piece=a.copy()
            else:
                dub.run_ffmpeg('-i',path,'-af',filt,'-ar',SR,'-c:a','pcm_f32le',converted)
                piece,sr=sf.read(converted,dtype='float64')
            gain_db=0.
            if g['group']=='26-27':
                # Match the existing C3 voice level without flattening three lines.
                reference=Path(read(ROOT/'reports/2026-10-22-dub-c3-expression/final-verification.json')['model_rows'][-1]['path'])
                previous,_=sf.read(reference,dtype='float64')
                gain_db=dub.speech_rms_db(previous,SR)-dub.speech_rms_db(piece,SR)
                piece*=10**(gain_db/20)
            if rate!=1.:piece=dub.fade(piece,SR,ms=4)
            dest=round(offset*SR);end=dest+len(piece)
            assert dest>=0 and end<=len(track)
            overlap=occupancy[dest:end]
            if np.any(overlap):
                old_peak=float(np.max(np.abs(track[dest:end][overlap])))
                new_peak=float(np.max(np.abs(piece[overlap])))
                overlaps.append(dict(group=g['group'],seconds=float(overlap.sum()/SR),old_peak=old_peak,new_peak=new_peak))
                assert min(old_peak,new_peak)<.01,overlaps[-1]
            track[dest:end]+=piece;occupancy[dest:end]=True
            group_path=target/'groups'/f"r2_g{g['group']}.wav";sf.write(group_path,piece,SR,subtype='PCM_24')
            rg={**g,'rendered_path':str(group_path),'rendered_sha256':sha(group_path),'start':dest/SR,'end':end/SR,'gain_db':float(gain_db),'internal_cuts':0,'inserted_internal_silence_seconds':0}
            rendered.append(rg)
            for i,(line,(a,b)) in enumerate(zip(g['lines'],g['source_cue_spans'])):
                start=offset+a/rate;stop=offset+b/rate
                rows.append(dict(index=len(rows)+1,group=g['group'],text=line['text'],synth=line['synth'],
                    window=[line['start'],line['end']],start=start,end=stop,rate=rate,
                    timestamp_method='Source word timestamps mapped through intact-group constant tempo; later ASR cross-check retained.',
                    tolerance=g['fit']['tolerance'],visual_window=g.get('visual_windows',[None]*len(g['lines']))[i]))
    assert len(rows)==55 and np.max(np.abs(track))<.99
    voice=target/'人声轨_48k24bit.wav';sf.write(voice,track,SR,subtype='PCM_24')
    result=dict(waveform_sha256=sha(voice),duration=178,sample_rate=SR,groups=rendered,rows=rows,
        group_count=24,cue_count=55,internal_cuts=0,added_internal_silence_seconds=0,overlaps=overlaps,
        source_code_changes=False,human_listening=False)
    write(target/'alignment.json',result)
    print('voice_rendered',sha(voice),'overlaps',overlaps,flush=True)

def mix_video():
    target=OUT/'aligned';alignment=read(target/'alignment.json');voicepath=target/'人声轨_48k24bit.wav'
    assert sha(voicepath)==alignment['waveform_sha256']
    video=base.VIDEO;musicpath=base.MUSIC;sfxpath=base.SFX
    assert sha(video)==base.EXPECTED_SOURCE_VIDEO_SHA256
    inputs={str(p):sha(p)for p in (voicepath,video,musicpath,sfxpath)}
    voice=base.read_track(voicepath);music=base.read_stem(musicpath,len(voice));sfx=base.read_stem(sfxpath,len(voice));bed=music+sfx
    mask=np.zeros(len(voice),dtype=bool)
    for r in alignment['rows']:mask[round(r['start']*SR):round(r['end']*SR)]=True
    # Restore the original soundtrack envelope.  No sentence pumping or tail duck.
    voice_gain_db=base.rms_db(bed,mask)+4-base.rms_db(voice,mask)
    voice_gain=10**(voice_gain_db/20)
    mixed_voice=voice*voice_gain
    # Keep the restored background intact; gently bring the whole closing take
    # forward instead of masking the music or patching individual spoken words.
    closing=alignment['groups'][-1]
    ca,cb=round(closing['start']*SR),round(closing['end']*SR)
    envelope=np.full(cb-ca,10**(2.5/20),dtype=np.float32)
    ramp=min(round(.06*SR),len(envelope)//2)
    envelope[:ramp]=np.linspace(1,envelope[0],ramp)
    envelope[-ramp:]=np.linspace(envelope[-1],1,ramp)
    mixed_voice[ca:cb]*=envelope
    combined=bed+mixed_voice[:,None]
    mixpath=OUT/'mix.wav'
    assert not mixpath.exists() and not FINAL.exists(),'Preserve prior deliverables'
    with tempfile.TemporaryDirectory(prefix='c3-r2-mix-')as temp:
        raw=Path(temp)/'raw.wav';sf.write(raw,combined,SR,subtype='FLOAT')
        loudness=base.loudnorm_two_pass(raw,mixpath)
    FINAL.parent.mkdir(parents=True,exist_ok=True)
    dub.run_ffmpeg('-i',video,'-i',mixpath,'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','256k','-ar',SR,'-ac',2,'-movflags','+faststart','-shortest',FINAL)
    validation=base.validate_final(FINAL,base.video_hash(video))
    stats=base.loudnorm_stats(FINAL)
    assert -16.5<=float(stats['input_i'])<=-15.5 and float(stats['input_tp'])<=-1.5,stats
    assert all(sha(p)==digest for p,digest in inputs.items())
    per_cue=[]
    for r in alignment['rows']:
        a,b=round(r['start']*SR),round(r['end']*SR)
        per_cue.append(dict(index=r['index'],group=r['group'],text=r['text'],voice_rms=base.rms_db(mixed_voice[a:b]),bed_rms=base.rms_db(bed[a:b])))
    result=dict(status='technical_checks_passed',video=str(FINAL),video_sha256=sha(FINAL),video_bytes=FINAL.stat().st_size,
        voice=str(voicepath),voice_sha256=sha(voicepath),input_sha256=inputs,validation=validation,
        soundtrack_policy='Original music and SFX unity gain before common output loudness normalization; no ducking; exact original timing.',
        voice_over_bed_db=4,voice_gain_db=voice_gain_db,closing_voice_presence_db=2.5,music_duck_db=0,sfx_duck_db=0,closing_extra_duck_db=0,
        loudness=loudness,final_aac_loudness=stats,per_cue=per_cue,human_listening=False)
    write(OUT/'mix-report.json',result);print(json.dumps({k:result[k]for k in ['video','video_bytes','final_aac_loudness']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['voice','mix']);a=p.parse_args()
    render_voice() if a.action=='voice' else mix_video()
