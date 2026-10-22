"""Align the approved C3 waveform and a newly approved-by-checks closing to V10 cues.

One-off orchestration only: no original script, text, or synthesis backend edits.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dub
from c_consistency_select import normalize_transcript as norm

OUT = ROOT / 'dub/out/c3-video-2026-10-22'
C3 = ROOT / 'dub/out/c3-expression-2026-10-22'
SOURCE_REPORT = ROOT / 'reports/2026-10-22-dub-c3-expression'
MEDIA = Path('D:/XiaoNian-Video/review/2026-10-21-v10')
SR = 48000
REPLACEMENTS = [('明瞭','明了'),('小念','晓念'),('小是让','晓是让'),
    ('要做的是','要做的事'),('原画','原话'),('原化','原话'),('代办','待办'),
    ('小黎','小狸'),('小离','小狸'),('他会','它会'),('在反馈','再反馈'),
    ('先际状态','先记状态'),('算是另外','算式另外'),('sqlight','sqlite')]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical_units(asr):
    """Keep timestamps while applying the existing explicit spelling rules."""
    tokens, times = asr['text'].split(), asr['timestamp']
    assert len(tokens) == len(times), 'Token/timestamp cardinality mismatch'
    chars, spans = [], []
    for token, (s, e) in zip(tokens, times):
        token = re.sub(r'[^\u4e00-\u9fffA-Za-z]', '', token).lower()
        for c in token:
            chars.append(c)
            spans.append((s/1000, e/1000))
    text = ''.join(chars)
    for old, new in REPLACEMENTS:
        at = 0
        while (i := text.find(old, at)) >= 0:
            old_spans = spans[i:i+len(old)]
            if len(old) == len(new):
                new_spans = old_spans
            else:
                # SQLite is one spoken term; no cue boundary lies inside it.
                new_spans = [(min(x[0] for x in old_spans), max(x[1] for x in old_spans))]*len(new)
            text = text[:i] + new + text[i+len(old):]
            spans[i:i+len(old)] = new_spans
            at = i+len(new)
    assert text == norm(asr['text'])
    return text, spans

def parse_srt():
    text = (MEDIA/'晓念_V10_字幕.srt').read_text(encoding='utf-8-sig')
    def seconds(t):
        h,m,s,ms = map(int,re.split('[:,]',t))
        return h*3600+m*60+s+ms/1000
    rows = []
    for block in re.split(r'\r?\n\s*\r?\n', text.strip()):
        lines = block.splitlines()
        a,b = lines[1].split(' --> ')
        rows.append(dict(index=int(lines[0]),start=seconds(a),end=seconds(b),text=''.join(lines[2:])))
    return rows

def valley(audio, a, b):
    """Choose a shared boundary, never duplicate the same phoneme on both sides."""
    center = (a+b)/2
    # With no ASR gap, limit the search to 15 ms either side of the estimate.
    lo, hi = (a,b) if b-a >= .02 else (center-.015,center+.015)
    lo_i,hi_i = max(0,round(lo*SR)),min(len(audio)-1,round(hi*SR))
    energy = uniform_filter1d(audio[max(0,lo_i-120):min(len(audio),hi_i+121)]**2,size=97)
    offset = max(0,lo_i-120)
    choices = energy[lo_i-offset:hi_i-offset+1]
    j = lo_i + int(np.argmin(choices))
    # Use a nearby zero crossing to reduce discontinuities without moving a word.
    l,r = max(lo_i,j-48),min(hi_i,j+48)
    signs = np.flatnonzero(np.signbit(audio[l:r]) != np.signbit(audio[l+1:r+1]))
    if len(signs):
        k = l+signs
        j = int(k[np.argmin(np.abs(k-j))])
    return j/SR

def align(draft=False):
    data = dub.load_script(dub.SCRIPT)
    verification = read(SOURCE_REPORT/'final-verification.json')
    full = C3/'delivery/C3_交流起伏_完整旁白.wav'
    assert sha(full) == verification['source_sha256'] == 'a0db0f1e86000b320f0d64d869c1e72122c89dacea3ba4bf3558ef530a4785b5'
    previous_plan = read(SOURCE_REPORT/'plan.json')
    for name,digest in previous_plan['source_hashes'].items():
        assert sha(ROOT/name) == digest, name
    srt = parse_srt()
    lines = [line for g in data['groups'] for line in g['lines']]
    assert len(lines)==55 and len(srt)==48
    differences=[]
    for line,sub in zip(lines[4:-3],srt):
        # Script rounds to centiseconds; use the source SRT's millisecond times.
        assert abs(line['start']-sub['start'])<=.0051 and abs(line['end']-sub['end'])<=.0051
        line['script_window']=[line['start'],line['end']]
        line['start'],line['end']=sub['start'],sub['end']
        if norm(line['text'])!=norm(sub['text']):
            differences.append(dict(srt_index=sub['index'],voice=line['text'],subtitle=sub['text'],start=line['start'],end=line['end']))
    target = OUT/('draft-aligned' if draft else 'aligned')
    target.mkdir(parents=True,exist_ok=True)
    (target/'cues').mkdir(exist_ok=True)
    sources = {r['group']:r for r in verification['model_rows']}
    closing = None if draft else read(OUT/'closing/selected.json')
    if closing:
        path = Path(closing['path'])
        assert sha(path)==closing['sha256']
        sources['26-27']=dict(path=str(path),sha256=closing['sha256'],paraformer=closing['paraformer'],
            sensevoice=closing.get('sensevoice'),new_closing=True)
    track = np.zeros(round(data['duration']*SR),dtype=np.float64)
    rows=[]
    all_boundaries=[]
    failure=[]
    for group in data['groups']:
        source = sources[group['id']]
        path=Path(source['path'])
        assert sha(path)==source['sha256']
        audio,sr=sf.read(path,dtype='float64')
        assert sr==SR and audio.ndim==1
        # New closing is leveled ONCE to C3's closing, never line by line.
        closing_gain_db = 0.0
        if source.get('new_closing'):
            old,_=sf.read(Path(verification['model_rows'][-1]['path']),dtype='float64')
            closing_gain_db = dub.speech_rms_db(old,SR)-dub.speech_rms_db(audio,SR)+1.0
            audio=audio*10**(closing_gain_db/20)
        canonical,spans=canonical_units(source['paraformer'])
        expected=''.join(norm(l['synth']) for l in group['lines'])
        mapping_disagreement=None
        if expected!=canonical and source.get('new_closing'):
            # The secondary recognizer verifies the complete original phrase.
            # Primary timestamps may still be used positionally for ONE known
            # character disagreement; this does not mark either transcript passed.
            mismatches=[i for i,(a,b) in enumerate(zip(expected,canonical)) if a!=b]
            assert len(expected)==len(canonical) and mismatches==[7]
            assert expected[7]=='晓' and canonical[7]=='想'
            assert norm(source['sensevoice']['plain'])==expected
            mapping_disagreement=dict(raw_primary=source['paraformer']['text'],
                secondary=source['sensevoice']['plain'],character_index=7,
                policy='Position-only timestamp mapping; final isolated cue ASR must independently pass.')
        else:
            assert expected==canonical,(group['id'],expected,canonical)
        cue_spans=[];offset=0
        for line in group['lines']:
            count=len(norm(line['synth']))
            cue_spans.append((spans[offset][0],spans[offset+count-1][1]))
            offset+=count
        boundaries=[]
        for i in range(len(cue_spans)-1):
            a,b=cue_spans[i][1],cue_spans[i+1][0]
            shared=valley(audio,a,b)
            boundaries.append(shared)
            all_boundaries.append(dict(group=group['id'],after_line=i+1,asr_gap=[a,b],cut=shared,short_gap=b-a<.12))
        for i,(line,(voice_a,voice_b)) in enumerate(zip(group['lines'],cue_spans)):
            lower=boundaries[i-1] if i else 0
            upper=boundaries[i] if i<len(boundaries) else len(audio)/SR
            start=max(lower,voice_a-.03)
            end=min(upper,voice_b+.06)
            # A shared no-gap boundary can move at most 15ms around ASR estimate.
            piece=audio[round(start*SR):round(end*SR)].copy()
            window=line['end']-line['start']
            speed=max(1.,(len(piece)/SR)/(window-.005))
            if speed>1.10001:
                failure.append(dict(group=group['id'],text=line['text'],speed=speed,reason='Over 1.10x needed'))
                continue
            source_piece=piece.copy()
            if speed>1.005:
                piece=dub.atempo(source_piece,SR,speed)
                # atempo has length rounding; retry from original, never truncate speech.
                if len(piece)>round(window*SR):
                    speed*=len(piece)/(window*SR)+.002
                    if speed>1.10001:
                        failure.append(dict(group=group['id'],text=line['text'],speed=speed,reason='Rendered overrun'))
                        continue
                    piece=dub.atempo(source_piece,SR,speed)
            else:
                speed=1.0
            assert len(piece)<=round(window*SR),(group['id'],line['text'],len(piece)/SR,window)
            # Only 2ms edges; no volume flattening or noise gate.
            piece=dub.fade(piece,SR,ms=2)
            dest=round(line['start']*SR)
            assert dest+len(piece)<=len(track)
            assert not np.any(track[dest:dest+len(piece)])
            track[dest:dest+len(piece)]=piece
            index=len(rows)+1
            cue_file=target/'cues'/f'cue_{index:02d}_g{group["id"]}.wav'
            sf.write(cue_file,piece,SR,subtype='PCM_24')
            rows.append(dict(index=index,group=group['id'],text=line['text'],synth=line['synth'],shot=line['shot'],
                window=[line['start'],line['end']],start=line['start'],end=line['start']+len(piece)/SR,
                speed=speed,source_path=str(path),source_sha256=source['sha256'],source_span=[start,end],
                original_asr_speech_span=[voice_a,voice_b],closing_gain_db=closing_gain_db,
                cue_path=str(cue_file),cue_sha256=sha(cue_file),new_closing=bool(source.get('new_closing'))))
            rows[-1]['source_asr_mapping_disagreement']=mapping_disagreement
    result=dict(approved_c3_path=str(full),approved_c3_sha256=sha(full),original_video=str(MEDIA/'晓念_V10_完整片_无旁白.mp4'),
        original_source_hashes=previous_plan['source_hashes'],source_srt_sha256=sha(MEDIA/'晓念_V10_字幕.srt'),
        duration=178,sample_rate=SR,cue_count=len(rows),rows=rows,boundaries=all_boundaries,failures=failure,
        subtitle_wording_differences=differences,subtitle_policy='Original video pixels and original subtitle words are preserved; voice uses unchanged approved script.',
        new_closing=not draft,synthetic_voice_generation_for_body=False)
    write(target/'alignment.json',result)
    if failure:raise SystemExit(json.dumps(failure,ensure_ascii=False))
    assert len(rows)==55
    peak=float(np.max(np.abs(track)))
    assert peak<.99,peak
    wave=target/'人声轨_48k24bit.wav';sf.write(wave,track,SR,subtype='PCM_24')
    result.update(waveform_sha256=sha(wave),peak_dbfs=float(20*np.log10(peak)),max_speed=max(r['speed']for r in rows))
    write(target/'alignment.json',result)
    print(json.dumps({k:v for k,v in result.items() if k in ['cue_count','waveform_sha256','max_speed','subtitle_wording_differences','new_closing']},ensure_ascii=False,indent=2))

def verify():
    folder=OUT/'aligned';report=read(folder/'alignment.json')
    wave=folder/'人声轨_48k24bit.wav'
    assert sha(wave)==report['waveform_sha256']
    audio,sr=sf.read(wave,dtype='float32')
    assert sr==SR and len(audio)==178*SR
    model=None
    cachefile=folder/'cue-asr-cache.json';cache=read(cachefile) if cachefile.exists() else {}
    rows=[];failed=[]
    for row in report['rows']:
        a=audio[round(row['start']*SR):round(row['end']*SR)]
        key=hashlib.sha256(a.tobytes()).hexdigest()
        if key not in cache:
            if model is None:
                from funasr import AutoModel
                model=AutoModel(model='paraformer-zh',device='cuda:0',ncpu=4,disable_update=True,disable_pbar=True)
            r=model.generate(input=resample_poly(a,16000,SR).astype(np.float32),fs=16000,disable_pbar=True)[0]
            cache[key]={k:v for k,v in r.items() if k in ['text','timestamp']}
            write(cachefile,cache)
        result=cache[key]
        checks=dict(text=norm(result['text'])==norm(row['synth']),start_exact=abs(row['start']-row['window'][0])<1/SR,
            end_inside=row['end']<=row['window'][1]+1/SR,tempo=row['speed']<=1.10001,non_silent=bool(np.max(np.abs(a))>.01))
        timing=result.get('timestamp',[])
        # Independent ASR on actual, edited final cues checks that no word was cut.
        r={**row,'asr':result,'checks':checks,'first_word_delay':timing[0][0]/1000 if timing else None,
            'last_word_absolute':row['start']+timing[-1][1]/1000 if timing else None}
        rows.append(r)
        if not all(checks.values()):failed.append(r['index'])
        print(f"cue {row['index']}/55 {'OK' if all(checks.values()) else 'REVIEW'} {row['text']} | {result['text']}",flush=True)
    result=dict(waveform_sha256=sha(wave),passed=not failed,failed_cues=failed,rows=rows,
        normalization='Existing explicit same-sound spelling rules only; raw text retained.',human_listening=False)
    write(folder/'cue-verification.json',result)
    if failed:raise SystemExit('Cue review needed: '+str(failed))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['draft','align','verify']);a=p.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    if a.action=='verify':verify()
    else:align(a.action=='draft')
