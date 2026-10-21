"""第二步 · 精调：围绕评分最高的候选，每次只改一个槽位，再加一组「统计最优组合」。

做什么：
  1. 原描述换几个新种子 —— 分清「这个方向好」还是「这一条运气好」
  2. 单变量微调 —— 年龄感/音调/质感/鼻音/语速，各换成其它选项试一遍
  3. 统计最优组合 —— 把每个槽位里平均分最高的选项拼在一起
  4. （可选）--custom 你自己手写的描述

例子：
  python step2_refine.py --model voxcpm2                         # 用 rounds/r1 的评分，取前 3 名
  python step2_refine.py --model voxcpm2 --top 5 --axes all
  python step2_refine.py --model voxcpm2 --ids r1_vox_3fa2c1,r1_vox_9b01de
  python step2_refine.py --model voxcpm2 --from rounds/r1 rounds/r2 --out rounds/r3   # 再精调一轮
  python step2_refine.py --model voxcpm2 --custom "十七岁的少女，声音软软的，……"
"""
from pathlib import Path

import specs
from common import (add_model_args, build_desc, clip_id, copy_clip, load_rounds, make_backend, new_parser,
                    run_jobs, slots_zh, top_rated)


def mean(xs):
    return sum(xs) / len(xs)


def best_combo(rated, fallback):
    """每个槽位取平均分最高（至少 2 条样本）的选项。样本太少时返回 None。"""
    with_slots = [e for e in rated if e.get("slots")]
    if len(with_slots) < 6:
        return None
    print("\n各槽位平均分（同模型、已评分）：")
    best = {}
    for axis in specs.SLOT_ORDER:
        groups = {}
        for e in with_slots:
            groups.setdefault(e["slots"][axis], []).append(e["score"])
        ranked = sorted(((mean(v), len(v), k) for k, v in groups.items()), reverse=True)
        print(f"  {specs.SLOT_LABELS[axis]}：" + "  ".join(
            f"{specs.SLOTS[axis][k][0]} {m:.2f}(n={n})" for m, n, k in ranked))
        solid = [r for r in ranked if r[1] >= 2]
        best[axis] = solid[0][2] if solid else fallback["slots"][axis]
    langs = {}
    for e in with_slots:
        langs.setdefault(e.get("desc_lang"), []).append(e["score"])
    lang_rank = sorted(((mean(v), len(v), k) for k, v in langs.items() if k in ("zh", "en")), reverse=True)
    print("  描述语言：" + "  ".join(f"{k} {m:.2f}(n={n})" for m, n, k in lang_rank))
    lang = lang_rank[0][2] if lang_rank else fallback.get("desc_lang", "zh")
    return best, lang


def main():
    parser = new_parser(__doc__)
    add_model_args(parser, default_out="rounds/r2")
    parser.add_argument("--from", dest="sources", nargs="+", default=["rounds/r1"], help="读取哪几轮的评分（默认 rounds/r1）")
    parser.add_argument("--top", type=int, default=3, help="取前几名做精调（默认 3）")
    parser.add_argument("--min-score", type=int, default=4, help="最低分数（默认 4）")
    parser.add_argument("--ids", help="直接指定要精调的候选 id（逗号分隔，在试音台关闭盲听后可以看到）")
    parser.add_argument("--axes", default="age,pitch,texture,nasal,pace",
                        help="单变量微调哪些槽位：age,pitch,texture,nasal,pace,mood,manner 或 all")
    parser.add_argument("--seeds", type=int, default=2, help="每个微调变体生成几个种子（默认 2）")
    parser.add_argument("--reseeds", type=int, default=4, help="原描述再换几个新种子（默认 4）")
    parser.add_argument("--no-best", action="store_true", help="不生成「统计最优组合」")
    parser.add_argument("--any-model", action="store_true", help="候选不限于当前模型（描述在不同模型上效果不同，一般不用）")
    parser.add_argument("--custom", action="append", default=[], help="自己写的描述，可写多次")
    parser.add_argument("--text", default=specs.AUDITION_LINE, help="试音台词（默认和海选相同，方便对比）")
    args = parser.parse_args()

    axes = specs.SLOT_ORDER if args.axes == "all" else [a.strip() for a in args.axes.split(",") if a.strip()]
    unknown = [a for a in axes if a not in specs.SLOTS]
    if unknown:
        raise SystemExit(f"不认识的槽位：{unknown}，可选 {specs.SLOT_ORDER}")

    entries = load_rounds(args.sources)
    pool = entries if args.any_model else [e for e in entries if e["model"] == args.model]
    if args.ids:
        wanted = [x.strip() for x in args.ids.split(",") if x.strip()]
        by_id = {e["id"]: e for e in entries}
        missing = [x for x in wanted if x not in by_id]
        if missing:
            raise SystemExit(f"找不到这些 id：{missing}")
        picks = [by_id[x] for x in wanted]
    else:
        picks = top_rated(pool, args.top, args.min_score)
        if not picks and not args.custom:
            raise SystemExit(f"{args.sources} 里没有 ≥{args.min_score} 分的 {args.model} 候选。"
                             "先用试音台评分，或用 --min-score 3 放宽，或用 --ids 指定。")

    round_name = Path(args.out).name
    reseeds = range(101, 101 + args.reseeds)
    vseeds = range(1, 1 + args.seeds)
    jobs = []

    def add(desc, seeds, **meta_extra):
        for seed in seeds:
            cid = clip_id(round_name, args.model, desc, args.text, seed)
            meta = dict(id=cid, file=f"{cid}.wav", round=round_name, model=args.model,
                        desc=desc, text=args.text, seed=seed, **meta_extra)
            jobs.append({"meta": meta, "run": lambda b, d=desc, s=seed: b.design(d, args.text, s)})

    print("本轮精调的候选：")
    for p in picks:
        print(f"  {p['id']}  ★{p['score']}  {p.get('archetype', '')}  {' '.join(p['tags'])}\n    {p['desc']}")
        base = dict(archetype=p.get("archetype", "自定义"), parent=p["id"], desc_lang=p.get("desc_lang"))
        slots = p.get("slots")
        if slots:
            base.update(slots=slots, slots_zh=slots_zh(slots))
        add(p["desc"], reseeds, variant="原描述·换种子", **base)
        if slots and p.get("desc_lang") in ("zh", "en"):
            for axis in axes:
                current = slots[axis]
                for value in specs.SLOTS[axis]:
                    if value == current:
                        continue
                    changed = {**slots, axis: value}
                    label = f"{specs.SLOT_LABELS[axis]}：{specs.SLOTS[axis][current][0]}→{specs.SLOTS[axis][value][0]}"
                    add(build_desc(changed, p["desc_lang"]), vseeds, variant=label,
                        **{**base, "slots": changed, "slots_zh": slots_zh(changed)})
        for tag in p["tags"]:
            if tag in specs.TAG_HINTS:
                axis, direction = specs.TAG_HINTS[tag]
                print(f"    提示：你标了「{tag}」，重点听「{specs.SLOT_LABELS[axis]}」往「{direction}」变的那几条")

    if picks and not args.no_best:
        combo = best_combo([e for e in pool if e["score"]], picks[0])
        if combo:
            slots, lang = combo
            desc = build_desc(slots, lang)
            if all(desc != p["desc"] for p in picks):
                print(f"\n统计最优组合 [{lang}]：{desc}")
                add(desc, range(1, 1 + args.seeds + args.reseeds), archetype="★统计最优组合", variant="统计最优组合",
                    desc_lang=lang, slots=slots, slots_zh=slots_zh(slots))
            else:
                print("\n统计最优组合和某个候选完全相同，已跳过。")

    for i, desc in enumerate(args.custom, 1):
        add(desc, range(1, 1 + args.reseeds), archetype=f"自定义{i}", variant="手写描述",
            desc_lang="custom")

    # 把精调起点也放进这一轮：盲听时它是对照组，关掉盲听后可以按 C/B 和变体做 A/B
    if not args.dry_run:
        for p in picks:
            copy_clip(p, args.out, "PARENT", variant="精调起点（上一轮原条）")

    print()
    run_jobs(jobs, args.out, lambda: make_backend(args, "design"), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
