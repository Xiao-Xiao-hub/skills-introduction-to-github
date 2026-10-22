"""Archive C3 only after actual waveform, ASR, and identity checks pass."""
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime

import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'dub/out/c3-expression-2026-10-22'
REPORT=ROOT/'reports/2026-10-22-dub-c3-expression'

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    plan=read(REPORT/'plan.json');verification=read(OUT/'audit-groups/final-verification.json')
    assembly=read(OUT/'natural-final/report.json');selection=read(OUT/'selection.json')
    speaker=read(OUT/'audit-groups-speaker-summary.json')['c3']
    from c3_identity import evaluate
    cache=read(OUT/'speaker-cache.json')
    rows=read(OUT/'audit-groups-speaker.json')
    identity=evaluate({r['group']:np.array(r['embedding']) for r in rows if r['set']=='audit-groups'},cache,plan)
    write(OUT/'final-identity.json',identity)
    wave=OUT/'natural-final/人声轨_48k24bit.wav'
    if not verification['technical_passed'] or sha(wave)!=verification['source_sha256']:
        raise SystemExit('Final waveform verification has not passed.')
    assert sum(bool(r.get('paraformer_match')) for r in verification['model_rows'])==24
    assert sum(r.get('sensevoice_plain') is not None and not r.get('needs_review') for r in verification['model_rows'])==5
    assert not verification.get('resolved_checks'), 'Current final must pass without historical short-clip resolution.'
    refs=list(identity['expression_prompt_scores'].values())
    identity_checks={'expression_prompt_min':min(refs)>=.85,'peer_median_min':min(identity['peer_medians'].values())>=.85,
                     'pairwise_p10':identity['pairwise_p10']>=.85,'pairwise_min':identity['pairwise_min']>=.80}
    if not all(identity_checks.values()):raise SystemExit('Final identity gate failed: '+str(identity_checks))
    if len(selection)!=24 or any(v['failed'] for v in selection.values()):raise SystemExit('Selection incomplete or failed.')
    delivery=OUT/'delivery';delivery.mkdir(exist_ok=True)
    finalname='C3_交流起伏_完整旁白.wav'
    shutil.copy2(wave,delivery/finalname);shutil.copy2(wave,REPORT/finalname)
    if sha(wave)!=sha(delivery/finalname) or sha(wave)!=sha(REPORT/finalname):raise SystemExit('Copy SHA mismatch')
    audio,sr=sf.read(wave,dtype='float64');chunks=[]
    for g in assembly['groups']:
        if g['group'] not in ['05-06','07','20','23a']:continue
        pos=g['actual_position'];chunks.extend([audio[round(pos['start']*sr):round(pos['end']*sr)],np.zeros(round(.7*sr))])
    preview='C3_重点片段试听.wav';sf.write(delivery/preview,np.concatenate(chunks),sr,subtype='PCM_24')
    shutil.copy2(delivery/preview,REPORT/preview)
    for folder in ['pilot','expressive-refs','lively-refs','takes']:
        for filename in ['manifest.jsonl','acoustic.json']:
            src=OUT/folder/filename
            if src.exists():
                dst=REPORT/'generation-evidence'/folder;dst.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst/filename)
    for src in [OUT/'selection.json',OUT/'candidate-review.json',OUT/'audit-groups/final-verification.json',OUT/'selected-identity.json',OUT/'final-identity.json',OUT/'rejected-trimmed-final-identity.json',OUT/'first-full-review-calm-reference.json',OUT/'rejected-first-final-verification.json',OUT/'identity-before-research-repair.json',
                OUT/'audit-groups-basic.json',OUT/'audit-groups-speaker-summary.json',
                OUT/'natural-final/report.json',OUT/'natural-final/配音报告.md',
                OUT/'pilot-asr.json',OUT/'basic-cache.meta.json',OUT/'speaker-cache.meta.json',OUT/'asr-cache.meta.json',
                OUT/'expressive-refs-asr.json',OUT/'lively-refs-asr.json',
                OUT/'expressive-refs-speaker-summary.json',OUT/'lively-refs-speaker-summary.json']:
        shutil.copy2(src,REPORT/src.name)
    history=REPORT/'prior-assembly';history.mkdir(exist_ok=True)
    for name in ['targeted-pronunciation.json','unresolved23b-final-verification.json','final-identity.json']:
        shutil.copy2(OUT/'before-preserve-edges'/name,history/name)
    shutil.copy2(OUT/'before-preserve-edges/audit-groups/final-verification.json',history/'final-verification.json')
    shutil.copy2(OUT/'before-preserve-edges/audit-groups/23b_长任务_最终音轨原句.wav',history/'23b_长任务_旧整轨原句.wav')
    chosen=REPORT/'selected-takes';chosen.mkdir(exist_ok=True)
    manifest=[]
    all_entries=[json.loads(x) for x in (OUT/'takes/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    for gid,entry in selection.items():
        src=OUT/'takes'/entry['file'];shutil.copy2(src,chosen/src.name)
        original=next(e for e in all_entries if e['file']==src.name)
        manifest.append({**original,'sha256':sha(src)})
    (chosen/'manifest.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in manifest),encoding='utf-8')
    snapshots=REPORT/'workflow-snapshots';snapshots.mkdir(exist_ok=True)
    for filename in ['c3_expression.py','c3_natural_assembly.py','c3_final_verify.py','c3_resolve_asr.py','c3_deliver.py','c3_identity.py','c_consistency_check.py','c_consistency_select.py']:
        shutil.copy2(ROOT/'deployment'/filename,snapshots/filename)
    total=sum(sum(not json.loads(line).get('reused_previous_take') for line in (OUT/folder/'manifest.jsonl').read_text(encoding='utf-8').splitlines()) for folder in ['pilot','expressive-refs','lively-refs','takes'])
    reused=[e for e in manifest if e.get('reused_previous_take')]
    result=dict(completed_at=datetime.now().astimezone().isoformat(),waveform_sha256=sha(wave),
                local_audio=str(delivery/finalname),local_preview=str(delivery/preview),bytes=wave.stat().st_size,
                duration=178,sample_rate=sr,channels=1,subtype='PCM_24',groups=24,original_cues=55,
                final_asr_groups_passed=24,secondary_whole_groups_matched=5,secondary_targeted_groups_matched=0,
                full_context_asr_disagreement_preserved=[],prior_wave_disagreement_preserved=['23b'],human_pronunciation_verified=False,
                synthesis_saved_candidates=total,selected_takes=24,reused_previous_takes=len(reused),identity_checks=identity_checks,
                identity_reference_min=min(refs),identity_pairwise_p10=speaker['pairwise_p10'],
                identity_pairwise_min=speaker['pairwise_min'],identity_pairwise_mean=speaker['pairwise_mean'],
                original_c2_calm_reference_min=min(identity['calm_reference_scores'].values()),
                naturalness_human_approved=False,exact_55_cue_alignment_verified=False,
                original_script_changes=False,backend_changes=False,video_created=False)
    import torch
    from importlib.metadata import version
    environment=dict(python=__import__('sys').version,torch=torch.__version__,cuda=torch.version.cuda,
                     cuda_available=torch.cuda.is_available(),capability=list(torch.cuda.get_device_capability()),
                     voxcpm=version('voxcpm'),funasr=version('funasr'),new_packages_installed=False)
    if not (environment['cuda_available'] and environment['cuda']=='12.8' and environment['capability']==[12,0]):
        raise SystemExit('Unexpected GPU environment')
    write(REPORT/'environment.json',environment)
    write(REPORT/'delivery.json',result);write(delivery/'delivery.json',result)
    plan.update(stage='technical_passed_user_listening_pending',completed_at=result['completed_at'],delivery=result)
    write(REPORT/'plan.json',plan)
    same_sound=[]
    for row in verification['model_rows']:
        if row['paraformer'].get('text')!=row['text']:
            same_sound.append({'group':row['group'],'expected':row['text'],'paraformer':row['paraformer'].get('text'),
                               'sensevoice':row.get('sensevoice_plain'),'normalized_match':row['paraformer_match']})
    write(REPORT/'raw-asr-spelling.json',same_sound)
    doc=f'''# C3 交流起伏完整旁白

用户听取C2后认为情绪不足、人机感重；试听C3第一版介绍小样后进一步要求“再多一些交流感和情绪起伏”。本轮交付完整178秒独立人声，保留55句原稿、24组内容；没有合成视频。

- 完整音频：`{finalname}`（48kHz、单声道、PCM24，{wave.stat().st_size:,}字节）。
- 快速试听：`{preview}`（介绍、提问、体谅、技术说明四段）。
- 本机位置：`{delivery/finalname}`。
- SHA256：`{sha(wave)}`。

## 表达与音频后期

保持一个身份参考，按内容选用温暖分享、体谅和正式说明的表达参考。用户反馈后重新制作了更有起伏的介绍/体谅参考并重做主体；选条强制核对当前参考哈希。事实和技术段保持克制。没有用“音高越平越好”筛选，也未进行人为音高拉齐。

本轮整组保留模型生成的语流和内部停顿，不再把每句切开按字幕重排。统一采用完整原take可放入组窗口即保留全部起音与收尾的规则；只有放不下时才裁外部静音并保留60/120ms保护量。音量先统一调整，再只修正偏离中位数超过2dB的组，保留表达的轻重差。按原视频24组的窗口落位，最多1.05倍整组变速；不截断内容。

## 已实际验证

- 最终波形：178秒、48kHz PCM24、无组间重叠、无越出组窗口或末尾截断，无满幅削波。
- Paraformer对实际最终波形提取的24组检查通过；SenseVoice对07、18、23a、23b、26-27五组全文归一化匹配，23b中的“长任务”在这次最终音轨的两种识别器全文结果中均一致。所有识别均未提供预期稿或热词。
- 识别中的空格、标点和已明确列出的同音异写归一后核对；原始结果保留于`final-verification.json`和`raw-asr-spelling.json`，不将其冒称全部字面完全相同。
- 最终各组相对其预先选定的表达参考最小余弦{min(refs):.3f}；组间余弦均值{speaker['pairwise_mean']:.3f}、P10 {speaker['pairwise_p10']:.3f}、最小{speaker['pairwise_min']:.3f}，每组对其它组的中位相似度也要求≥0.85。这些只支持身份漂移检查，不能证明听感自然。
- 与C2旧calm基准的最小相似度为{min(identity['calm_reference_scores'].values()):.3f}，逐组数值透明保留在`final-identity.json`；本轮不声称精确匹配C2旧音色。旧calm单参考门槛会将新情感表达也一起淘汰，因此改为预选表达参考保真与全片跨组一致性联合门槛；首轮全片min0.773/P10.831未通过，已据此补录，未直接放低门槛使它过关。
- specs.py、dub/xiaonian_v10.json、common.py及所有原脚本哈希不变；没有改Backend、安装新模型或flash-attn。全部模型串行运行。

## 需要注意

- 本版优先保留段落内部自然语流；24组与视频段落窗口对齐，55条字幕逐句精确同步尚未验收。正式合视频时需要按这条人声复核字幕与画面。
- 技术检查通过，情绪、真人感、音色衔接与参赛适用性仍需用户试听；没有人工自然度通过记录，也不以声纹或音高指标替代听感。
- 22组取本轮素材，2组明确复用C2已复核读音的原始素材：收尾26-27的5条新候选中“晓”持续被识别为“想”；技术段23b新seed41仍被SenseVoice识别为“场任务”，故采用已复核的seed43。新整轨上的这两段再次做双ASR和声音一致性检查。未将复用素材冒称新生成。
- 先前裁切版的“长任务”存在全文识别分歧、短句复核通过；该版的原始识别、短句与SHA保存在`prior-assembly/`，只作历史，不作为当前音轨验收证据。当前音轨的两种全文识别均为“长任务”；“算是另外”等仍通过声明的同音规则归一，不能等同于人耳确认。
- 首轮整轨裁去首尾静音后，个别相似度在门槛边缘降到0.795和0.849；已保存失败记录，统一保留可容纳的完整原take后重新检查实际成片，门槛未放宽。此举使部分组的语音起点较裁切版延后0.06–0.30秒，仍位于原段落窗口内。
- 原始合成报告的“未运行ASR”描述只覆盖后期合成步骤；后续实际最终音频ASR结论见`final-verification.json`。

## 过程与证据

共保存{total}条新候选，最终使用其中{24-len(reused)}条并复用{len(reused)}条C2素材。初始两条错误地将控制词放入prompt续写模式而读出指令，已全部淘汰；一轮表达参考发生身份漂移，已淘汰其中20与23a。用户要求增强起伏后，保留旧候选并生成新的表达参考及整片。没有覆盖旧C/C2、评分或原稿。

`expression-bank/`保存本轮四类参考（最终分配见plan，早期question参考保留但未采用），`selected-takes/`保存最终采用的24条原始音频及manifest；其它候选元数据在`generation-evidence/`，原始未选音频仍留在本机。`workflow-snapshots/`是本轮一次性编排与后期助手的快照；恢复运行应放回工具包`deployment/`，它们复用原工具函数而未修改原脚本。

官方依据（采集2026-10-22）：[VoxCPM app模式互斥](https://github.com/OpenBMB/VoxCPM/blob/main/app.py)、[参考与prompt接口](https://github.com/OpenBMB/VoxCPM/blob/main/src/voxcpm/core.py)。
'''
    (REPORT/'README.md').write_text(doc,encoding='utf-8')
    with (ROOT/'PROGRESS.md').open('a',encoding='utf-8')as f:
        f.write(f'\n\n## 2026-10-22 C3 交流起伏版完成技术检查\n\n按用户“更多交流感和情绪起伏”反馈重做，交付178秒/48kHz/PCM24独立人声，24组完整；最终音频24组主ASR和5组交叉ASR全文归一化匹配；此前裁切版23b识别分歧已留历史证据，未混用为当前验收。对预选表达参考最小相似度{min(refs):.3f}，保留组内自然停顿与组间轻重变化。不声称精确复刻旧C2。原稿/脚本/Backend未改。人听自然度与55字幕逐句同步未验收，未合视频。完整音频及证据：`reports/2026-10-22-dub-c3-expression/`。\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
