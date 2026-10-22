"""C3 one-off orchestration. Existing toolkit, script, and backends stay unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dub
from common import VoxCPM2Backend, run_jobs, read_manifest

OUT = ROOT / 'dub/out/c3-expression-2026-10-22'
REPORT = ROOT / 'reports/2026-10-22-dub-c3-expression'
OLD = ROOT / 'reports/2026-10-22-dub-c-consistent/plan.json'
INTENTS = {
    '01-02': '想起一个熟悉的小瞬间，带一点理解的笑意，第二句轻轻转折。',
    '03': '像在回想自己的经历，念头冒出来时轻一点，散去时轻轻落下。',
    '04': '真诚地分享一个简单办法，睡前略停，先写下来轻轻强调。',
    '05-06': '带轻微笑意介绍自己的作品，开头自然引出，列举连贯，最后一句有邀请感。',
    '07': '前一句自然回顾，后一句是关心对方的真实提问，问句有轻微起伏。',
    '08': '完成待办时轻快一点，说到留下念头时放松，前后有自然转折。',
    '09': '像面对面解释一个方便之处，记录清楚，最后一句轻松落下。',
    '10': '像和观众一起操作，先看和再把稍作强调，节奏连贯。',
    '11': '清楚自然地演示两个动作，前后连着说，结尾带一点邀请感。',
    '12': '分组比较略紧凑，找到新联系时有一点发现感，最后一句笃定收住。',
    '13': '第一句带一点思索，第二句自然转为有办法的口吻。',
    '14': '认真而自然地解释依据，突出自己试算和两组结果的区别，句尾有轻重变化。',
    '15': '清楚地解释实现，计算核验略强调，第二句接着说明。',
    '16': '耐心地陪人找出问题，定位出错步骤略强调，收进错题本自然收住。',
    '17': '带一点亲近的笑意介绍小狸，追问和反馈层次清楚。',
    '18': '自然地引导下一步操作，校对原文稍强调，后半句连贯。',
    '19': '温和鼓励对方独立尝试，独立做一道稍强调，结尾轻松笃定。',
    '20': '真诚体谅，语气温暖舒展，像认真听对方说话，最后一句轻轻鼓励。',
    '21': '认真克制地说明必要提醒，现实中的求助清楚强调，句尾落稳。',
    '22': '带一点实现目标后的笃定，前句逐渐展开，后句自然收束。',
    '23a': '像开发者自信地向评委解释实际实现，语意连贯，技术名词清楚。',
    '23b': '接着解释实现，各个步骤有层次，关键动作稍强调，语气自然笃定。',
    '24': '自信地讲清楚系统如何工作，列举连贯，最后一句稳稳收住。',
    '26-27': '真诚舒展地说出作品的含义，两句各有重心，最后的名字温暖而笃定。',
}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def get_plan():
    path = REPORT/'plan.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    previous = json.loads(OLD.read_text(encoding='utf-8'))
    refs = {p['id']:p for p in previous['profiles']}
    profiles = []
    for pid, ref, base in [
        ('warm', 'calm', '沉稳温暖的女声，自然地向听众介绍自己的作品，语气真诚，轻重有变化。'),
        ('engaged', 'original', '成熟自然的女声，像面对面讲解自己的作品，有交流感，语句连贯，重点明确。'),
    ]:
        p = dict(refs[ref])
        p.update(id=pid, base_control=base, gen={'cfg':2.0,'steps':20})
        profiles.append(p)
    plan = dict(output=str(OUT.relative_to(ROOT)), user_feedback='C2情感不丰富、人机感重；平衡音色一致和自然表达',
                profiles=profiles, intents=INTENTS, source_hashes=previous['source_hashes'],
                model='VoxCPM2', mode='prompt+reference+control', selected_profile=None,
                sources=['https://github.com/OpenBMB/VoxCPM/blob/main/app.py',
                         'https://github.com/OpenBMB/VoxCPM/blob/main/src/voxcpm/core.py'],
                rules=['原稿与所有原脚本保持不变','同一模型串行生成','不以狭窄音高范围筛掉正常情感变化',
                       '说话人相似度仅检查身份漂移，不能证明自然度','保留原始ASR和所有候选，不伪造试听评分'])
    write(path, plan)
    return plan


def generate(args):
    plan = get_plan()
    data = dub.load_script(dub.SCRIPT)
    groups = dub.pick_groups(data, None if args.groups=='all' else args.groups)
    jobs = []
    for p in plan['profiles']:
        if args.profile!='all' and p['id']!=args.profile: continue
        if hashlib.sha256((ROOT/p['reference']).read_bytes()).hexdigest()!=p['sha256']:
            raise SystemExit('Reference hash mismatch')
        for g in groups:
            control=p['base_control']+INTENTS[g['id']]
            for seed in map(int,args.seeds.split(',')):
                key=[p['id'],g['text'],control,seed,p['gen'],p['sha256'],'reference-only']
                digest=hashlib.sha256(json.dumps(key,ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:8]
                cid=f"{p['id']}_g{g['id']}_s{seed}_{digest}"
                meta=dict(id=cid,file=cid+'.wav',model='voxcpm2',mode='reference+control',
                          profile=p['id'],ref=p['reference'],reference_sha256=p['sha256'],reference_text=p['reference_text'],
                          group=g['id'],text=g['text'],style=control,seed=seed,take=seed,gen=p['gen'],
                          gen_label='cfg2.0 步数20',desc='固定参考与分段表达指令',archetype=p['id'],variant=g['id'])
                jobs.append(dict(meta=meta,run=lambda b,p=p,t=g['text'],s=seed,c=control:
                    b.clone(ROOT/p['reference'],p['reference_text'],t,s,p['gen'],c)))
    run_jobs(jobs,OUT/args.folder,lambda:VoxCPM2Backend(str(ROOT/'models/VoxCPM2')))


def generate_full(args):
    plan=get_plan()
    identity=next(p for p in plan['profiles'] if p['id']=='warm')
    groups=dub.pick_groups(dub.load_script(dub.SCRIPT),None if args.groups=='all' else args.groups)
    jobs=[]
    for g in groups:
        anchor=plan['expression_bank'][plan['group_bank'][g['id']]]
        prompt=ROOT/anchor['path']
        if hashlib.sha256(prompt.read_bytes()).hexdigest()!=anchor['sha256']:raise SystemExit('Expression prompt hash mismatch')
        for seed in map(int,args.seeds.split(',')):
            key=[g['text'],seed,identity['gen'],identity['sha256'],anchor['sha256']]
            digest=hashlib.sha256(json.dumps(key,ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:8]
            cid=f"c3_g{g['id']}_s{seed}_{digest}"
            meta=dict(id=cid,file=cid+'.wav',model='voxcpm2',mode='prompt+reference',profile='c3',
                      group=g['id'],text=g['text'],style='',intent=INTENTS[g['id']],
                      ref=identity['reference'],reference_sha256=identity['sha256'],reference_text=identity['reference_text'],
                      prompt_path=anchor['path'],prompt_text=anchor['text'],prompt_sha256=anchor['sha256'],
                      expression_bank=plan['group_bank'][g['id']],seed=seed,take=seed,gen=identity['gen'],
                      gen_label='cfg2.0 步数20',desc='统一音色参考与按内容选择的表达参考',archetype='c3',variant=g['id'])
            jobs.append(dict(meta=meta,run=lambda b,t=g['text'],s=seed,a=anchor:
                b._generate(s,identity['gen'],text=t,prompt_wav_path=str(ROOT/a['path']),prompt_text=a['text'],
                            reference_wav_path=str(ROOT/identity['reference']))))
    run_jobs(jobs,OUT/args.folder,lambda:VoxCPM2Backend(str(ROOT/'models/VoxCPM2')))


def select(args):
    from c_consistency_select import normalize_transcript
    from c3_natural_assembly import trim_outer_silence
    import time
    active_selection=OUT/'selection.json'
    if active_selection.exists():active_selection.replace(OUT/f'selection.previous-{time.time_ns()}.json')
    plan=get_plan();identity=next(p for p in plan['profiles'] if p['id']=='warm')
    basic=json.loads((OUT/'basic-cache.json').read_text(encoding='utf-8'))
    speaker=json.loads((OUT/'speaker-cache.json').read_text(encoding='utf-8'))
    asr=json.loads((OUT/'asr-cache.json').read_text(encoding='utf-8'))
    ref=np.array(speaker[identity['sha256']]['embedding'])
    groups=dub.load_script(dub.SCRIPT)['groups'];entries=read_manifest(OUT/'takes')
    selection={};review=[];failed=[]
    for i,g in enumerate(groups):
        following=groups[i+1]['lines'][0]['start'] if i+1<len(groups) else 178
        available=min(g['lines'][-1]['end']+.35,following-.06)-g['lines'][0]['start']
        candidates=[]
        for e in entries:
            if e['group']!=g['id']:continue
            path=OUT/'takes'/e['file'];sha=hashlib.sha256(path.read_bytes()).hexdigest()
            if any(sha not in c for c in [basic,speaker,asr]):continue
            a,sr=sf.read(path);on=np.flatnonzero(dub.voiced(a,sr));hop=int(sr*dub.FRAME)
            start=max(0,int(on[0]*hop)-int(.06*sr)) if len(on) else 0
            end=min(len(a),int((on[-1]+1)*hop)+int(.12*sr)) if len(on) else len(a)
            natural_piece,_=trim_outer_silence(a,sr)
            need=(len(natural_piece)/sr-.18)/(available-.18)
            vec=np.array(speaker[sha]['embedding'])
            sim=float(vec@ref)
            prompt_sim=float(vec@np.array(speaker[e['prompt_sha256']]['embedding']))
            checks=dict(text=e['text']==g['text'],reference=e['reference_sha256']==identity['sha256'],
                        current_expression_bank=e.get('prompt_sha256')==plan['expression_bank'][plan['group_bank'][g['id']]]['sha256'],
                        no_spoken_control=e['style']=='',transcript=normalize_transcript(g['text'])==normalize_transcript(asr[sha]['text']),
                        duration=need<=1.05,speaker=prompt_sim>=.85,non_silent=bool(len(on)),sample_rate=sr==48000,
                        not_rejected=sha not in plan.get('rejected_take_sha256',{}))
            row=dict(file=e['file'],sha256=sha,seed=e['seed'],reference_sha256=e['reference_sha256'],gen=e['gen'],
                     prompt_sha256=e['prompt_sha256'],expression_bank=e['expression_bank'],speaker_cosine=sim,expression_prompt_cosine=prompt_sim,
                     need=need,asr=asr[sha]['text'],expected=g['text'],checks=checks,
                     failed=[k for k,v in checks.items() if not v],basic=basic[sha])
            # Do not reward flat pitch or choose a different emotion just for higher cosine.
            row['rank']=[len(row['failed']),max(0,need-1),-prompt_sim]
            candidates.append(row)
        candidates.sort(key=lambda r:r['rank'])
        if not candidates:failed.append(g['id']);continue
        best=candidates[0];selection[g['id']]=best
        review.append(dict(group=g['id'],selected=best,candidates=candidates))
        if best['failed']:failed.append(g['id'])
        print(g['id'],best['seed'],f"speaker={best['speaker_cosine']:.3f} need={best['need']:.3f}",best['failed'],
              best['asr'] if not best['checks']['transcript'] else '',flush=True)
    write(OUT/'candidate-review.json',dict(groups=review,failed_groups=failed))
    if failed:raise SystemExit('RETRY '+','.join(failed))
    from c3_identity import evaluate
    result=evaluate({g:np.array(speaker[s['sha256']]['embedding']) for g,s in selection.items()},speaker,plan)
    write(OUT/'selected-identity.json',result)
    if not result['passed']:raise SystemExit('IDENTITY_RETRY '+json.dumps(result['failures'],ensure_ascii=False))
    pending=OUT/'selection.pending.json'
    write(pending,selection);pending.replace(active_selection)


def check(args):
    import c_consistency_check as checks
    checks.OUT=OUT; checks.REPORT=REPORT
    def records(folder):
        rows=[dict(path=p['reference'],text=p['reference_text'],set='reference',profile=p['id'],group='reference') for p in get_plan()['profiles']]
        for e in read_manifest(OUT/folder):
            rows.append(dict(path=str(OUT/folder/e['file']),text=e['text'],set=folder,profile=e['profile'],group=e['group']))
        return rows
    checks.records=records
    sys.argv=['c3',args.kind,'--folder',args.folder]
    checks.main()


def preview(args):
    entries=read_manifest(OUT/args.folder)
    target=OUT/'preview'; target.mkdir(parents=True,exist_ok=True)
    for profile in dict.fromkeys(e['profile'] for e in entries):
        chunks=[]
        for e in entries:
            if e['profile']!=profile:continue
            a,s=sf.read(OUT/args.folder/e['file'],dtype='float64')
            gain=10**((-20-dub.speech_rms_db(a,s))/20)
            a=dub.fade(a*min(gain,.89/max(np.max(np.abs(a)),1e-12)),s)
            chunks.extend([a,np.zeros(int(.7*s))])
        sf.write(target/f'{profile}_表达小样.wav',np.concatenate(chunks),s,subtype='PCM_24')
        print(target/f'{profile}_表达小样.wav')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['generate','full','check','preview','select'])
    parser.add_argument('--profile',default='all');parser.add_argument('--groups',default='05-06,07,20,23a')
    parser.add_argument('--seeds',default='41');parser.add_argument('--folder',default='pilot')
    parser.add_argument('--kind',choices=['basic','speaker','asr'],default='basic')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);get_plan()
    {'generate':generate,'full':generate_full,'check':check,'preview':preview,'select':select}[args.action](args)
