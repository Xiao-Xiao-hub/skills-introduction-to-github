"""Fit intact C3 phrase groups to visible subtitle windows, retaining natural joins."""
import numpy as np
from c3_video_align import ROOT, read, write, norm, canonical_units, parse_srt, dub

OUT=ROOT/'dub/out/c3-video-r2-2026-10-22'

def source_rows():
    original=read(ROOT/'reports/2026-10-22-dub-c3-expression/final-verification.json')
    return {r['group']:r for r in original['model_rows']}

def plan():
    data=dub.load_script(dub.SCRIPT)
    lines=[l for g in data['groups']for l in g['lines']]
    for line,sub in zip(lines[4:-3],parse_srt()):line['start'],line['end']=sub['start'],sub['end']
    sources=source_rows();rows=[]
    for g in data['groups'][:-1]:
        source=sources[g['id']];canonical,spans=canonical_units(source['paraformer'])
        assert canonical==''.join(norm(l['synth'])for l in g['lines'])
        cue_spans=[];offset=0
        for l in g['lines']:
            count=len(norm(l['synth']));cue_spans.append((spans[offset][0],spans[offset+count-1][1]));offset+=count
        candidates=[]
        for tolerance in (0,.04,.08,.12,.16):
            for rate in np.arange(.80,1.0401,.001):
                # Only internal boundaries may have a small cross-caption margin.
                lower=max(l['start']-(tolerance if i else 0)-a/rate for i,(l,(a,b)) in enumerate(zip(g['lines'],cue_spans)))
                upper=min(l['end']+(tolerance if i<len(g['lines'])-1 else 0)-b/rate for i,(l,(a,b)) in enumerate(zip(g['lines'],cue_spans)))
                if lower>upper:continue
                # Start as early as constraints permit; prefer gentle slowing.
                first_delay=lower+cue_spans[0][0]/rate-g['lines'][0]['start']
                score=abs(rate-.93)*2+.35*first_delay+max(0,first_delay-.8)*3
                candidates.append(dict(rate=float(rate),offset=float(lower),first_voice_delay=float(first_delay),tolerance=tolerance,score=float(score)))
            if candidates:break
        item=dict(group=g['id'],source=source,lines=g['lines'],source_cue_spans=cue_spans)
        if candidates:
            best=min(candidates,key=lambda x:x['score'])
            best['actual_cue_spans']=[[best['offset']+a/best['rate'],best['offset']+b/best['rate']]for a,b in cue_spans]
            item['fit']=best
            if g['id']=='23b':
                item['fit']=dict(rate=1.,offset=141.19,first_voice_delay=.13,tolerance=.02,
                    reason='Keep approved original wave untouched; 0.3 percent tempo change was unnecessary and introduced recognition disagreement.')
        else:item['fit']=None
        rows.append(item)
    write(OUT/'body-group-plan.json',dict(groups=rows,method='Intact approved C3 groups, one constant tempo per group, no internal cuts or inserted silence. ASR estimates constrain speech to source subtitles.'))
    for r in rows:
        f=r['fit'];print(r['group'],None if f is None else {k:round(v,4)for k,v in f.items()if k in ['rate','offset','first_voice_delay','tolerance']})
    return rows

if __name__=='__main__':plan()
