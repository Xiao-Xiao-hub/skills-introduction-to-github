"""Publish anonymous checks and new-text synthesized audio, never private corpus."""
from pathlib import Path
import sys,json,collections,shutil,subprocess
import split_voice_scenes_20261024 as S
ROOT=S.ROOT;BASE=S.BASE;REPORT=ROOT/'reports/2026-10-24-separated-registers-v2'

def audit():
    sets={scene:{s:S.rows(BASE/scene/f'{s}_raw.jsonl')for s in ('train','val','test')}for scene in ('formal','daily')}
    allrows={scene:sum(parts.values(),[])for scene,parts in sets.items()}
    assert {r['source_id']for r in allrows['formal']}.isdisjoint({r['source_id']for r in allrows['daily']})
    for scene,parts in sets.items():
        assert len({r['id']for r in allrows[scene]})==len(allrows[scene])
        assert all(S.sha(r['audio'])==r['audio_sha256']for r in allrows[scene])
        if scene=='daily':
            for a,b in [('train','val'),('train','test'),('val','test')]:assert {r['split_group']for r in parts[a]}.isdisjoint({r['split_group']for r in parts[b]})
    annotations=S.read(BASE/'annotations.json')['rows'];formal_names=sorted({r.get('source_original_name')for r in annotations if r['accepted']and r['scene']=='formal'})
    assert formal_names==['1-1.m4a','1-6.m4a']
    protected=['specs.py','common.py','step1_audition.py','dub.py','dub/xiaonian_v10.json']
    diff=subprocess.check_output(['git','diff','HEAD','--',*protected],cwd=ROOT,text=True);assert not diff
    daily=[r for r in annotations if r['accepted']and r['scene']=='daily']
    candidates=[r for r in annotations if r['scene']=='daily']
    S.write(REPORT/'daily-curation-audit.json',dict(candidates=len(candidates),accepted=len(daily),rejected=len(candidates)-len(daily),exclusion_reason_counts=dict(collections.Counter(reason for r in candidates if not r['accepted']for reason in r['exclusion_reasons'])),accepted_audio_variants=dict(collections.Counter(r['variant']for r in daily)),native_bandwidth_originals=sum(r.get('native_24k_source',False)for r in daily),human_audio_reviewed_count=0,complete_verbatim_annotations_kept_private=True,quality_gates=dict(SIG_min=3.2,BAK_min=3.3,OVRL_min=2.95)))
    style=dict(scene='daily',source='genuine chat voice only',word_counts=S.read(BASE/'daily/selection-summary.json')['word_counts'],repetitions_not_removed=True,self_repairs_not_rewritten=True,accent_inferred_from_spelling=False,emotion_candidates=dict(collections.Counter(r['emotion']for r in daily)),emotion_labels_human_verified=False,explicit_emotion_conditioning_trained=False,tts_does_not_automatically_rewrite_written_language=True,human_audio_reviewed_count=0)
    S.write(BASE/'daily/style-profile.json',style)
    proof=dict(formal_sources=formal_names,formal_and_daily_source_disjoint=True,all_audio_hashes_match=True,daily_date_groups_disjoint=True,formal_holdout_independent_sessions=False,formal_and_daily_audio_hashes_disjoint={r['audio_sha256']for r in allrows['formal']}.isdisjoint({r['audio_sha256']for r in allrows['daily']}),protected_original_scripts_unchanged=True,reference_hashes={s:S.read(BASE/s/'reference.json')['sha256']for s in sets},dataset_snapshots={s:S.sha(BASE/s/'snapshot.json')for s in sets},synthetic_training_audio_count=0,human_audio_reviewed_count=0)
    S.write(BASE/'separation-audit.json',proof);REPORT.mkdir(exist_ok=True);S.write(REPORT/'separation-audit.json',proof);S.write(REPORT/'daily-style-profile.json',style)
    for scene in sets:
        for name in ('selection-summary.json','training-result.json','alignment-proof.json','validation-history.jsonl','sampling.json'):
            p=BASE/scene/name
            if p.exists():shutil.copy2(p,REPORT/(scene+'-'+name))
    print('AUDIT_PASS',json.dumps({k:len(v)for k,v in allrows.items()}),flush=True)

def finish():
    audit();evaluation=S.read(BASE/'audition/evaluation.json');dest=REPORT/'audition';dest.mkdir(exist_ok=True);out=[]
    for r in evaluation['rows']:
        public=dict(r)
        for key in ('path','preview'):
            p=Path(r[key]);shutil.copy2(p,dest/p.name);public[key]='audition/'+p.name
        out.append(public)
    S.write(REPORT/'audition-checks.json',dict(rows=out,human_listening_accepted=False,asr_and_pitch_are_not_naturalness_acceptance=True))
    proof=BASE/'audition/adapter-reload-proof.json'
    if proof.exists():shutil.copy2(proof,REPORT/proof.name)
    snapshots=REPORT/'workflow-snapshots';snapshots.mkdir(exist_ok=True);files=['split_voice_scenes_20261024.py','train_scene_qwen_20261024.py','scene_voice_runtime_20261024.py','audition_scene_qwen_20261024.py','report_scene_qwen_20261024.py']
    for name in files:shutil.copy2(ROOT/'deployment'/name,snapshots/name)
    shutil.copy2(Path('E:/Qwen3-TTS-training/LICENSE'),snapshots/'UPSTREAM-LICENSE')
    S.write(snapshots/'manifest.json',dict(files={n:S.sha(snapshots/n)for n in files},dependencies='deployment helpers preserved in previous qwen-exclusive-voice workflow snapshot; upstream Qwen trainer pinned at 022e286b98fbec7e1e916cb940cdf532cd9f488e',original_toolkit_logic_modified=False))
    results={s:S.read(BASE/s/'training-result.json')for s in ('formal','daily')}
    assert all(r['weight_update_proof']['all_finite']and r['weight_update_proof']['changed']==462 for r in results.values())
    assert results['formal']['adapter_sha256']!=results['daily']['adapter_sha256']
    S.write(REPORT/'completion.json',dict(completed=True,separate_adapters=True,initialization='original Base independently for each scene',adapter_sha256={s:r['adapter_sha256']for s,r in results.items()},synthetic_audition_count=len(out),human_listening_accepted=False,private_data_and_weights_uploaded=False))
    print('REPORT_COMPLETE',len(out),flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
