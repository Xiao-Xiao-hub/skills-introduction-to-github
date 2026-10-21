"""第一步 · 海选：10 个方向 × 中/英文描述 × 若干种子，批量生成候选音色。

例子：
  python step1_audition.py --model voxcpm2              # VoxCPM2 环境里跑
  python step1_audition.py --model qwen3                # Qwen3-TTS 环境里跑（输出到同一个文件夹）
  python step1_audition.py --model voxcpm2 --smoke      # 部署后冒烟测试：只生成 1 条
  python step1_audition.py --model voxcpm2 --only A,D --seeds 5
  python step1_audition.py --model voxcpm2 --dry-run    # 只看会生成什么
"""
from pathlib import Path

import specs
from common import add_model_args, build_desc, clip_id, make_backend, new_parser, run_jobs, slots_zh


def main():
    parser = new_parser(__doc__)
    add_model_args(parser, default_out="rounds/r1")
    parser.add_argument("--seeds", type=int, default=3, help="每条描述生成几个种子（默认 3）")
    parser.add_argument("--seed-start", type=int, default=1, help="起始种子（默认 1）")
    parser.add_argument("--langs", default="zh,en", help="描述语言：zh、en 或 zh,en（默认两种都试）")
    parser.add_argument("--only", help="只跑某几个方向，如 A,D,G")
    parser.add_argument("--text", default=specs.AUDITION_LINE, help="试音台词（默认用 specs.py 里的）")
    parser.add_argument("--smoke", action="store_true", help="冒烟测试：方向 A、中文描述、1 个种子，输出到 rounds/smoke")
    args = parser.parse_args()

    archetypes = specs.ARCHETYPES
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    if args.smoke:
        archetypes, langs, seeds, args.out = archetypes[:1], ["zh"], [args.seed_start], "rounds/smoke"
    elif args.only:
        wanted = {x.strip().upper() for x in args.only.split(",")}
        archetypes = [a for a in archetypes if a[0] in wanted]
    if not archetypes or not langs or not seeds:
        raise SystemExit("没有要生成的内容，检查 --only / --langs / --seeds")

    round_name = Path(args.out).name
    jobs = []
    for code, name, slots in archetypes:
        for lang in langs:
            desc = build_desc(slots, lang)
            for seed in seeds:
                cid = clip_id(round_name, args.model, desc, args.text, seed)
                meta = dict(id=cid, file=f"{cid}.wav", round=round_name, model=args.model,
                            archetype=f"{code}{name}", slots=slots, slots_zh=slots_zh(slots),
                            desc_lang=lang, desc=desc, text=args.text, seed=seed)
                jobs.append({"meta": meta, "run": lambda b, d=desc, s=seed: b.design(d, args.text, s)})

    if args.dry_run:
        print("本轮的描述：")
        for code, name, slots in archetypes:
            for lang in langs:
                print(f"  {code}{name} [{lang}] {build_desc(slots, lang)}")
    run_jobs(jobs, args.out, lambda: make_backend(args, "design"), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
