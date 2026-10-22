"""Final machine acceptance gate; human naturalness is explicitly not inferred."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import dub

OUT=ROOT/'dub/out/c-consistent-2026-10-22'
REPORT=ROOT/'reports/2026-10-22-dub-c-consistent'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
plan=read(REPORT/'plan.json');wave=read(OUT/'final-waveform-verification.json')
speaker=read(OUT/'audit-groups-speaker-summary.json')
basics=read(OUT/'audit-groups-basic.json')
ref=next(r for r in basics if r['set']=='reference' and r['profile']=='calm')
order=[g['id'] for g in dub.load_script(dub.SCRIPT)['groups']]

def metrics(profile):
    rows=[r for r in basics if r['profile']==profile]
    rows.sort(key=lambda r:order.index(r['group']))
    if len(rows)!=24:raise SystemExit(f'{profile}: missing final groups')
    mean=np.array([r['acoustic']['f0'] for r in rows],dtype=float)
    med=np.array([r['f0_median'] for r in rows])
    st=12*np.log2(mean)
    sim=speaker[profile]
    return dict(group_count=len(rows),mean_f0_min=float(mean.min()),mean_f0_max=float(mean.max()),
        mean_f0_average=float(mean.mean()),mean_f0_sd=float(mean.std()),mean_f0_cv=float(mean.std()/mean.mean()),
        mean_f0_p90_p10_semitones=float(np.percentile(st,90)-np.percentile(st,10)),
        mean_f0_adjacent_max_semitones=float(np.abs(np.diff(st)).max()),
        median_f0_anchor_max_abs_semitones=float(np.abs(12*np.log2(med/ref['f0_median'])).max()),
        speaker_reference_min=min(sim['reference_scores']['calm']),speaker_reference_mean=float(np.mean(sim['reference_scores']['calm'])),
        speaker_pairwise_min=sim['pairwise_min'],speaker_pairwise_p10=sim['pairwise_p10'],speaker_pairwise_mean=sim['pairwise_mean'],
        hnr_min=min(r['acoustic']['hnr'] for r in rows),rate_min=min(r['acoustic']['rate'] for r in rows),
        rate_max=max(r['acoustic']['rate'] for r in rows),intonation_range_min=min(r['acoustic']['range_st'] for r in rows),
        group_values=[{'group':r['group'],'mean_f0':r['acoustic']['f0'],'f0_median':r['f0_median'],
                      'hnr':r['acoustic']['hnr'],'rate':r['acoustic']['rate'],'range_st':r['acoustic']['range_st']} for r in rows])

new=metrics('final');old=metrics('old_final')
t=plan['acceptance_targets']
checks={'content_and_timing':wave['content_timing_checks_passed'],
        'speaker_reference':new['speaker_reference_min']>=t['speaker_reference_min'],
        'speaker_pairwise_p10':new['speaker_pairwise_p10']>=t['speaker_pairwise_p10_min'],
        'speaker_pairwise_min':new['speaker_pairwise_min']>=t['speaker_pairwise_min'],
        'pitch_to_anchor':new['median_f0_anchor_max_abs_semitones']<=t['f0_median_from_anchor_max_semitones'],
        'pitch_distribution':new['mean_f0_p90_p10_semitones']<=t['f0_mean_p90_p10_max_semitones'],
        'adjacent_pitch':new['mean_f0_adjacent_max_semitones']<=t['adjacent_mean_f0_max_semitones'],
        'hnr':new['hnr_min']>=t['hnr_min_db'],'rate':new['rate_max']<=t['speech_rate_max_char_per_second'],
        'preserved_intonation':new['intonation_range_min']>=5,
        'current_waveform_hash':hashlib.sha256((OUT/'final/人声轨_48k24bit.wav').read_bytes()).hexdigest()==wave['waveform_sha256']}
cross=read(OUT/'cross-final-asr.json')
checks['independent_key_cue_asr']=cross['all_key_cues_passed'] and cross['waveform_sha256']==wave['waveform_sha256']
source={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha for name,sha in plan['source_hashes'].items()}
checks['source_unchanged']=all(source.values())
result={'machine_checks_passed':all(checks.values()),'checks':checks,'old_C_final':old,'new_C_final':new,
        'waveform_sha256':wave['waveform_sha256'],'criteria_source':'Predeclared task-specific engineering thresholds in plan.json; not validated perceptual quality scores.',
        'human_listening_passed':False,'human_listening_status':'User listening required; no human reviewer claimed.',
        'content_timing_report':'final-waveform-verification.json','source_hash_checks':source}
(OUT/'acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['old_C_final','new_C_final','source_hash_checks']},ensure_ascii=False,indent=2))
print('OLD',json.dumps({k:v for k,v in old.items() if k!='group_values'},ensure_ascii=False))
print('NEW',json.dumps({k:v for k,v in new.items() if k!='group_values'},ensure_ascii=False))
if not result['machine_checks_passed']:raise SystemExit(1)
