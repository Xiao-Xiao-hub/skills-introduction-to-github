"""第三步 · 克隆压力测试：拿最终候选当参考音频，让模型用这个声音说 6 句完全不同的话。

挑声音时只听了一句台词；以后真正配音靠的是「克隆」。这一步检验：
换了台词以后音色稳不稳、忧郁和可爱能不能都演出来、长句和中英混读会不会崩。

例子：
  python step3_clone_test.py --model voxcpm2 --from rounds/r1 rounds/r2            # 取所有轮次里的前 4 名
  python step3_clone_test.py --model voxcpm2 --from rounds/r2 --ids r2_vox_1a2b3c,r2_vox_4d5e6f
  python step3_clone_test.py --model qwen3 --from rounds/r1 rounds/r2              # 用 Qwen3 Base 克隆（无语气控制）
"""
from pathlib import Path

import specs
from common import (add_model_args, clip_id, copy_clip, gen_grid, gen_label, load_rounds, make_backend, new_parser,
                    run_jobs, top_rated)


def main():
    parser = new_parser(__doc__)
    add_model_args(parser, default_out="rounds/clone")
    parser.add_argument("--from", dest="sources", nargs="+", default=["rounds/r1", "rounds/r2"],
                        help="从哪几轮挑参考音频（默认 rounds/r1 rounds/r2）")
    parser.add_argument("--top", type=int, default=4, help="取前几名（默认 4）")
    parser.add_argument("--min-score", type=int, default=4, help="最低分数（默认 4）")
    parser.add_argument("--ids", help="直接指定参考候选的 id（逗号分隔）")
    parser.add_argument("--seeds", type=int, default=1, help="每句生成几个种子（默认 1）")
    args = parser.parse_args()
    if args.model == "moss":
        raise SystemExit("MOSS-VoiceGenerator 只能造声，克隆测试请用 --model voxcpm2 或 qwen3")

    sources = [s for s in args.sources if Path(s).is_dir()]
    if not sources:
        raise SystemExit(f"找不到 {args.sources}")
    entries = load_rounds(sources)
    if args.ids:
        by_id = {e["id"]: e for e in entries}
        wanted = [x.strip() for x in args.ids.split(",") if x.strip()]
        missing = [x for x in wanted if x not in by_id]
        if missing:
            raise SystemExit(f"找不到这些 id：{missing}")
        refs = [by_id[x] for x in wanted]
    else:
        refs = top_rated(entries, args.top, args.min_score, key=lambda e: e["id"])
    if not refs:
        raise SystemExit(f"没有 ≥{args.min_score} 分的候选，先评分或用 --ids 指定。")

    out = Path(args.out)
    round_name = out.name
    lines = specs.CLONE_TEST_LINES
    if args.model == "qwen3":
        print("提示：Qwen3 Base 克隆不支持语气控制，「忧郁」「可爱」两句会按参考音频的语气读。")

    jobs = []
    print("参考音频：")
    for ref in refs:
        ref_path = Path(ref["dir"]) / ref["file"]
        print(f"  {ref['id']}  ★{ref['score']}  {ref.get('archetype', '')}  {ref.get('variant') or ''}")
        # 命令行没指定生成参数时，同一个模型的参考沿用它自己的参数
        gens = gen_grid(args, inherit=ref.get("gen") if ref.get("model") == args.model else None)
        for label, style, text in lines:
            for gen in gens:
                for seed in range(1, 1 + args.seeds):
                    cid = clip_id(round_name, args.model, "clone", ref["id"], text, style, seed, gen)
                    meta = dict(id=cid, file=f"{cid}.wav", round=round_name, model=args.model, parent=ref["id"],
                                archetype=ref.get("archetype"), line=label, style=style, text=text, seed=seed,
                                gen=gen, gen_label=gen_label(gen), ref_desc=ref.get("desc"))
                    jobs.append({"meta": meta, "run": lambda b, p=ref_path, rt=ref["text"], t=text, st=style, s=seed, g=gen:
                                 b.clone(p, rt, t, s, g, st)})

    # 把参考原声也放进来，试音台里可以和克隆结果对照着听
    if not args.dry_run:
        for ref in refs:
            copy_clip(ref, out, "REF", line="参考原声")

    run_jobs(jobs, out, lambda: make_backend(args, "clone"), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
