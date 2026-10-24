"""Finalize trained voice evidence and verified synthetic auditions only."""
from pathlib import Path
import json,shutil
import prepare_qwen_voice_20261024 as D
ROOT=D.ROOT;OUT=D.OUT;REPORT=ROOT/'reports/2026-10-24-qwen-exclusive-voice'

def main():
    training=D.read(OUT/'training-result.json');export=D.read(OUT/'export-result.json');summary=D.read(OUT/'selection-summary.json');evaluation=D.read(OUT/'audition/evaluation.json')['rows'];index={r['id']:r for r in evaluation}
    assert training['completed'] and training['weight_update_proof']['all_finite'] and training['weight_update_proof']['changed_tensors']>0
    preservation=D.read(OUT/'preservation-before.json');after={k:dict(v,unchanged=D.sha(v['path'])==v['sha256'])for k,v in preservation.items()};assert all(r['unchanged']for r in after.values());D.write(OUT/'preservation-after.json',after)
    audio=REPORT/'audition';audio.mkdir(exist_ok=True);exports=[]
    labels={'daily':'日常交流','happy':'开心回应','warm':'温和解释','thoughtful':'认真思考','continuity':'连续语流'}
    for kind,label in labels.items():
        for mode,prefix in [('trained','专属模型'),('base','训练前对照')]:
            r=index[f'{mode}_{kind}']
            if r['exact_asr_matches']<2 or r['clipped_samples']:continue
            name=f'{prefix}_{label}.wav';dest=audio/name;shutil.copy2(r['preview'],dest)
            exports.append(dict(file='audition/'+name,id=r['id'],text=r['text'],seconds=r['duration'],sha256=D.sha(dest),human_accepted=False))
    for key in ('trained_clone_warm','reloaded_adapter_warm'):
        r=index.get(key)
        if r and r['exact_asr_matches']>=2 and not r['clipped_samples']:
            dest=audio/(key+'.wav');shutil.copy2(r['preview'],dest);exports.append(dict(file='audition/'+dest.name,id=key,text=r['text'],seconds=r['duration'],sha256=D.sha(dest),human_accepted=False))
    fields=('id','mode','kind','text','duration','seconds','sha256','reference_sha256','checkpoint_sha256','reference_audio_required_at_inference','generation_method','seed','temperature','top_p','clipped_samples','longest_internal_low_energy_gap_s','internal_gaps_over_0_6s','f0_median_hz','f0_p10_hz','f0_p90_hz','formal_reference_cosine','chat_centroid_cosine','exact_asr_matches','continuity_windows','minimum_window_cosine')
    checks=[]
    for r in evaluation:
        item={k:r[k]for k in fields if k in r}
        item['asr']={n:{k:r[n][k]for k in ('match','cer')}for n in ('para','sense','qwen')}
        if r['mode']=='trained':
            base=index.get('base_'+r['kind']);item['duration_ratio_to_base']=r['duration']/base['duration'];item['large_tempo_change_flag']=not .65<=item['duration_ratio_to_base']<=1.6
        checks.append(item)
    D.write(REPORT/'audition-checks.json',dict(rows=checks,exports=exports,transcripts_are_checks_not_naturalness_acceptance=True,human_listening_accepted=False,normalization='constant gain only, -23dBFS RMS and peak limit .85; no EQ/time edits'))
    for name in ('training-result.json','export-result.json','validation-history.jsonl','selection-summary.json','alignment-proof.json'):
        shutil.copy2(OUT/name,REPORT/name)
    D.write(REPORT/'preservation-audit.json',dict(all_anchors_unchanged=True,anchors={k:r['sha256']for k,r in after.items()},training_snapshot_sha256=D.sha(OUT/'snapshot.json')))
    for name in ('baseline-reproduction.json','novel-text-check.json'):
        if (OUT/'audition'/name).exists():shutil.copy2(OUT/'audition'/name,REPORT/name)
    code=REPORT/'workflow-snapshots';code.mkdir(exist_ok=True)
    files=['prepare_qwen_voice_20261024.py','train_qwen_voice_20261024.py','audition_qwen_voice_20261024.py','use_qwen_xuan_20261024.py','package_qwen_voice_20261024.py']
    hashes={}
    for name in files:shutil.copy2(ROOT/'deployment'/name,code/name);hashes[name]=D.sha(code/name)
    shutil.copy2(Path('E:/Qwen3-TTS-training/LICENSE'),code/'UPSTREAM-LICENSE')
    D.write(code/'manifest.json',dict(sha256=hashes,note='Exact local deployment helper snapshots for review. Require existing local corpus helper dependencies and ROOT placement; not a standalone public training corpus.'))
    localcode=OUT/'repro-code';localcode.mkdir(exist_ok=True)
    for name in files:shutil.copy2(ROOT/'deployment'/name,localcode/name)
    D.write(ROOT/'voices/human-a-v1/qwen-experiment-v1.json',dict(id='xuan-qwen-exclusive-v1',status='trained_pending_user_listening',model_path='models/Qwen3-TTS-Base',adapter_path=str(Path(export['canonical_adapter_package']).relative_to(ROOT)),speaker='xuan',adapter_checkpoint=str(Path(training['selected_checkpoint']).relative_to(ROOT)),adapter_sha256=training['checkpoint_sha256'],diagnostic_merged_weight_sha256=export['exported_weight_sha256'],merged_model_selected=False,reference_audio_required_at_inference=False,training=summary['splits'],accepted_reference_clone_baseline='reports/2026-10-24-qq-voice-v2/clarity-diagnosis/03_另一模型_未降噪日常参考.wav',report='reports/2026-10-24-qwen-exclusive-voice/README.md',human_listening_accepted=False))
    before=training['base_test']['total'];after_loss=training['selected_test']['total'];improvement=(before-after_loss)/before*100
    audition_rows='\n'.join(f"| [{r['id']}]({r['file']}) | {r['seconds']:.2f}秒 |"for r in exports)
    text=f'''

## 修正后的实训完成与交付

真实训练完成{training['epochs']}轮、{training['optimizer_steps']}个优化步，耗时{training['elapsed_s']:.1f}秒；峰值分配显存{training['peak_allocated_gib']:.3f}GiB。只按验证选第{training['selected_epoch']}轮，验证损失{training['initial_validation']['total']:.6f}→{training['best_validation']:.6f}。独立9条测试总损失{before:.6f}→{after_loss:.6f}（相对变化{improvement:.2f}%）；逐项数据在training-result.json。该指标不等于真人感、情绪或音色通过。

适配器SHA256：{training['checkpoint_sha256']}。{training['weight_update_proof']['tensors']}个LoRA张量中{training['weight_update_proof']['changed_tensors']}个变化，全部有限。这是真实更新模型权重，固定参考的说话人embedding同时作为专属speaker=xuan保存；没有只提取声纹冒称训练。

默认采用专属适配器包：本机`{export['canonical_adapter_package']}`，与既有Qwen3-TTS Base共同加载，使用已保存的speaker=xuan生成，无需参考录音。入口为deployment/use_qwen_xuan_20261024.py，reloaded_adapter_warm是全新进程从该目录重新加载的实际检查。

合并BF16诊断模型保留在`{export['path']}`，未选为默认：它的日常短句时长11.68秒、句内低能量停顿2.1秒，较基准4.64秒明显变长。未经合并的专属适配器同句5.60秒、最长间隔0.88秒。合并计算改变浮点舍入后，自回归采样可能变化；这是本机观察，不能单凭一个样本归因所有模型。诊断音不作为主试听交付，完整检查记录保留。默认个人适配器和原模型均保留，不宣称已通过用户听感。

新台词包括日常交流、开心、温和解释、思考与连续语流；专属模型使用CustomVoice，基准使用用户第三条相同参考/seed/温度的Base克隆，另留适配器+参考克隆诊断。三路ASR、削波、句内低能量停顿、F0及CAM数值在audition-checks.json；只公开核对通过的新台词合成音，异常转写不公开以防泄露训练原句。机器检查不证明自然感。

| 试听 | 时长 |
|---|---|
{audition_rows}

72段/6.93分钟是严格筛选后的总量，训练为5.24分钟；这次为排除疑点而减少总量，新增录用3段。没有声称素材越多越好、全面覆盖情绪或已人工精标。逐段文字/情绪/处理/取舍表在本机`voices/human-a-v1/local/qwen-corpus-20261024-v1/annotations.tsv`。

基础Qwen权重、原A配置、原第三条试听和上轮语料快照哈希均未变。本轮保持试听阶段，没有生成完整视频或重新启用被否决的Vox权重。
'''
    with (REPORT/'README.md').open('a',encoding='utf-8')as f:f.write(text)
    with (ROOT/'PROGRESS.md').open('a',encoding='utf-8')as f:f.write(f'\n\n## 2026-10-24 Qwen专属权重训练与导出完成，等待试听\n\n修正上游训练对齐错误后从基础模型重训8轮/{training["optimizer_steps"]}步，按验证选择第{training["selected_epoch"]}轮，独立测试损失变化{improvement:.2f}%。已导出speaker=xuan的专属模型，实际重新加载生成检查；基准与新模型的多语气新台词已核对。仍待用户听感验收，未做全片。[训练与试听](reports/2026-10-24-qwen-exclusive-voice/README.md)。\n')
    print(json.dumps(dict(exports=exports,improvement_percent=improvement),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
