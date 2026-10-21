"""声学测量：给一轮的每条音频算出音高、语调起伏、停顿占比、语速、气声程度，写进 acoustic.json。

每轮生成结束后会自动运行；也可以手动补测：
  python measure.py rounds/r1
  python measure.py rounds/r1 rounds/r2

试音台关掉盲听后会显示这些数字，最高分列表里也有。数字只用来辅助判断「为什么听着像机器 / 像小孩」，
最终还是以耳朵为准。需要：pip install praat-parselmouth
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

try:
    import parselmouth
    from parselmouth.praat import call
except ImportError:
    parselmouth = None

ACOUSTIC = "acoustic.json"

# 粗略参考线（按第一次试听的两条样本和普通话说话的一般情况定的，不是硬标准）
FLAGS = [
    ("range_st", lambda v: v < 5, "语调偏平"),
    ("pause_pct", lambda v: v > 25, "停顿偏多"),
    ("rate", lambda v: v < 3.5, "语速偏慢"),
    ("rate", lambda v: v > 6.5, "语速偏快"),
    ("f0", lambda v: v > 270, "音高偏高"),
    ("hnr", lambda v: v < 10, "气声偏重"),
]


def available():
    return parselmouth is not None


def analyze(path, text):
    snd = parselmouth.Sound(str(path))
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=120, pitch_ceiling=650)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    if len(f0) < 10:
        return None
    semitones = 12 * np.log2(f0 / 100.0)

    # 用响度找出说话区间，统计其中 ≥0.2 秒的停顿
    loud = snd.to_intensity(minimum_pitch=120, time_step=0.01).values[0]
    voiced = loud > loud.max() - 25
    idx = np.where(voiced)[0]
    start, end = idx[0], idx[-1]
    pauses, run = [], 0
    for v in voiced[start:end + 1]:
        if v:
            if run >= 20:
                pauses.append(run)
            run = 0
        else:
            run += 1
    span = (end - start) * 0.01
    pause = sum(pauses) * 0.01
    syllables = len(re.findall(r"[一-鿿]", text or ""))

    harm = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=120).values[0]
    harm = harm[harm > -50]

    result = {
        "f0": round(float(f0.mean())),                                   # 平均音高 Hz，越高越显小
        "range_st": round(float(np.percentile(semitones, 90) - np.percentile(semitones, 10)), 1),  # 语调起伏（半音）
        "pause_pct": round(100 * pause / span) if span else 0,           # 停顿占说话时长的百分比
        "rate": round(syllables / max(span - pause, 0.1), 1) if syllables else None,  # 每秒字数（不含停顿）
        "hnr": round(float(harm.mean()), 1) if len(harm) else None,      # 谐噪比 dB，越低气声越重
    }
    result["flags"] = [name for key, test, name in FLAGS if result.get(key) is not None and test(result[key])]
    return result


def describe(m):
    text = f"音高 {m['f0']}Hz · 起伏 {m['range_st']} 半音 · 停顿 {m['pause_pct']}%"
    if m.get("rate"):
        text += f" · 语速 {m['rate']} 字/秒"
    if m.get("hnr") is not None:
        text += f" · 谐噪比 {m['hnr']}dB"
    if m["flags"]:
        text += "  ⚠" + "、".join(m["flags"])
    return text


def measure_folder(folder, verbose=False):
    folder = Path(folder)
    out = folder / ACOUSTIC
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    manifest = folder / "manifest.jsonl"
    entries = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip()] if manifest.exists() else []
    new = 0
    for e in entries:
        wav = folder / e["file"]
        if e["file"] in data or not wav.exists():
            continue
        try:
            result = analyze(wav, e.get("text"))
        except Exception as exc:  # 个别文件测不出来不影响其它
            print(f"  测量失败 {e['file']}：{exc}")
            continue
        if result:
            data[e["file"]] = result
            new += 1
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    flagged = {}
    for m in data.values():
        for f in m["flags"]:
            flagged[f] = flagged.get(f, 0) + 1
    summary = "、".join(f"{k} {v} 条" for k, v in sorted(flagged.items(), key=lambda kv: -kv[1])) or "无"
    print(f"声学测量：新测 {new} 条，共 {len(data)} 条 → {out}。提示项：{summary}")
    if verbose:
        meta = {e["file"]: e for e in entries}
        for name, m in sorted(data.items(), key=lambda kv: (len(kv[1]["flags"]), -kv[1]["range_st"])):
            label = meta.get(name, {}).get("archetype", "")
            print(f"  {name}  {label}  {describe(m)}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+", help="要测量的轮次文件夹")
    args = parser.parse_args()
    if not available():
        raise SystemExit("需要先安装：pip install praat-parselmouth")
    for folder in args.folders:
        measure_folder(folder, verbose=True)


if __name__ == "__main__":
    main()
