"""One-off orchestration; uses unchanged toolkit generation APIs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dub
from common import VoxCPM2Backend, run_jobs

parser = argparse.ArgumentParser()
parser.add_argument('--profile', default='all')
parser.add_argument('--groups', default='07,16,20')
parser.add_argument('--seeds', default='41')
parser.add_argument('--out', default='pilot')
args = parser.parse_args()
plan = json.loads((ROOT/'reports/2026-10-22-dub-c-consistent/plan.json').read_text(encoding='utf-8'))
data = dub.load_script(dub.SCRIPT)
groups = dub.pick_groups(data, None if args.groups == 'all' else args.groups)
jobs = []
for profile in plan['profiles']:
    if args.profile != 'all' and profile['id'] != args.profile:
        continue
    for group in groups:
        for seed in map(int, args.seeds.split(',')):
            key = [profile['id'], group['text'], seed, profile['gen'], profile['sha256']]
            digest = hashlib.sha256(json.dumps(key, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:8]
            cid = f"{profile['id']}_g{group['id']}_s{seed}_{digest}"
            meta = dict(id=cid, file=cid+'.wav', model='voxcpm2', mode='clone',
                        profile=profile['id'], ref=profile['reference'],
                        reference_sha256=profile['sha256'], reference_text=profile['reference_text'],
                        group=group['id'], text=group['text'], style='', seed=seed, take=seed,
                        gen=profile['gen'], gen_label=f"cfg{profile['gen']['cfg']} 步数{profile['gen']['steps']}", desc='固定参考的高保真克隆',
                        archetype=profile['id'], variant=group['id'])
            jobs.append(dict(meta=meta, run=lambda b,p=profile,t=group['text'],s=seed:
                b.clone(ROOT/p['reference'],p['reference_text'],t,s,p['gen'],'')))
run_jobs(jobs, ROOT/plan['output']/args.out, lambda: VoxCPM2Backend(str(ROOT/'models/VoxCPM2')))
