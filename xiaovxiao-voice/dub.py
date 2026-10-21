"""给视频配音：用选定的哓v哓音色，把配音稿分组生成、按句切开、落到字幕时间上，再和原片声音混音、合成视频。

用 VoxCPM2（xvx-vox 环境）：它能「克隆音色 + 用一句话控制语气」。

  1. 试配：几个候选音色各配几段，用试音台盲听选一个
     python dub.py test --refs r0b_vox_821713,r0b_vox_49a14f,r0b_moss_611767,r0_vox_959950
  2. 正式配音：每组生成 3 条，自动挑最合适的，拼成整条人声轨，混音并合成视频
     python dub.py render --ref r0b_vox_821713 --video 成片.mp4
  3. 修改：用试音台打开 dub/out/<音色>/takes，给不满意的组重新打分（同组里分最高的那条会被选用），然后
     python dub.py assemble --ref r0b_vox_821713 --video 成片.mp4
     或者给某几组多生成几条再挑：python dub.py render --ref r0b_vox_821713 --only 05-06,14 --takes 5 --video 成片.mp4

需要 ffmpeg：系统里有就用系统的，没有就 pip install imageio-ffmpeg。
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from common import HERE, VoxCPM2Backend, read_manifest, read_ratings, run_jobs

SCRIPT = HERE / "dub" / "xiaonian_v10.json"
OUT_ROOT = HERE / "dub" / "out"
MODE_LABELS = {"style": "语气控制", "clone": "高保真克隆"}
FRAME = 0.01  # 静音分析的帧长（秒）


# ---------- 稿子与参考音色 ----------

def load_script(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    subs = data.get("substitutions", {})
    for group in data["groups"]:
        for line in group["lines"]:
            synth = line["text"]
            for old, new in subs.items():
                synth = synth.replace(old, new)
            line["synth"] = synth
        group["text"] = "".join(line["synth"] for line in group["lines"])
    return data


def pick_groups(data, only):
    if not only:
        return data["groups"]
    wanted = [x.strip() for x in only.split(",") if x.strip()]
    by_id = {g["id"]: g for g in data["groups"]}
    missing = [x for x in wanted if x not in by_id]
    if missing:
        raise SystemExit(f"没有这些组：{missing}。可选：{', '.join(by_id)}")
    return [by_id[x] for x in wanted]


def find_clip(ref, ref_text=None):
    """参考音色可以是候选 id（在 reports/ 或 rounds/ 的清单里找），也可以直接是 wav 路径。"""
    path = Path(ref)
    if path.suffix.lower() == ".wav" and path.exists():
        if not ref_text:
            raise SystemExit("直接给 wav 路径时，要用 --ref-text 写上这段音频说的内容")
        return path, {"id": path.stem, "text": ref_text}
    for manifest in sorted(HERE.glob("reports/*/manifest.jsonl")) + sorted(HERE.glob("rounds/*/manifest.jsonl")):
        for entry in read_manifest(manifest.parent):
            if entry["id"] == ref and (manifest.parent / entry["file"]).exists():
                return manifest.parent / entry["file"], entry
    raise SystemExit(f"找不到参考音色 {ref}（reports/ 和 rounds/ 的 manifest.jsonl 里都没有）")


def take_jobs(data, groups, ref, mode, seeds, gen, label_by_ref=False):
    ref_path, ref_entry = find_clip(ref)
    jobs = []
    for group in groups:
        style = f"{data['base_style']}，{group['style']}" if mode == "style" else ""
        for seed in seeds:
            key = json.dumps([ref, mode, group["text"], style, seed, gen], ensure_ascii=False, sort_keys=True)
            digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:4]
            cid = f"{ref}_{mode}_{group['id']}_{digest}" if label_by_ref else f"g{group['id']}_t{seed}_{digest}"
            meta = dict(id=cid, file=f"{cid}.wav", model="voxcpm2", ref=ref, mode=mode, group=group["id"], take=seed,
                        text=group["text"], style=style, desc=style or f"高保真克隆 {ref}", seed=seed, gen=gen,
                        gen_label=f"cfg{gen['cfg']:g} 步数{gen['steps']}")
            if label_by_ref:  # 试配：按音色、方式、段落分组统计
                meta.update(archetype=ref, variant=MODE_LABELS[mode], line=group["id"])
            else:
                meta.update(archetype=f"组 {group['id']}", variant=f"第{seed}条")
            jobs.append({"meta": meta, "run": lambda b, t=group["text"], st=style, s=seed:
                         b.clone(ref_path, ref_entry["text"], t, s, gen, st)})
    return jobs


# ---------- 音频分析 ----------

def frame_db(audio, sr):
    hop = int(sr * FRAME)
    n = max(1, len(audio) // hop)
    frames = audio[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    return 20 * np.log10(rms / (rms.max() + 1e-12))


def runs(mask, value):
    out, start = [], None
    for i, v in enumerate(mask):
        if v == value and start is None:
            start = i
        elif v != value and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def voiced(audio, sr, thresh=-40.0):
    mask = frame_db(audio, sr) > thresh
    for s, e in runs(mask, False):  # 填掉 50ms 以内的小空隙
        if s > 0 and e < len(mask) and e - s <= 5:
            mask[s:e] = True
    for s, e in runs(mask, True):  # 去掉 30ms 以内的孤立响声
        if e - s <= 3:
            mask[s:e] = False
    return mask


def text_weight(text):
    weight = len(re.findall(r"[一-鿿]", text)) + 0.5 * len(re.findall(r"[A-Za-z]", text))
    weight += 0.3 * text.count("、") + 0.6 * sum(text.count(c) for c in "，,；;")
    return weight + 1.0 * sum(text.count(c) for c in "。？！?!")


def split_lines(audio, sr, texts):
    """按句切开：在静音处找切点，切点位置要接近按字数估算的位置。切不出来返回 None。"""
    mask = voiced(audio, sr)
    on = np.where(mask)[0]
    if len(on) == 0:
        return None
    first, last = int(on[0]), int(on[-1]) + 1
    if len(texts) == 1:
        return [(first, last)]
    gaps = [(s, e) for s, e in runs(mask, False) if s > first and e < last and e - s >= 8]
    k, n = len(texts) - 1, len(gaps)
    if n < k:
        return None
    span = last - first
    weights = [text_weight(t) for t in texts]
    total = sum(weights)
    expect = [first + span * sum(weights[: j + 1]) / total for j in range(k)]
    mids = [(s + e) / 2 for s, e in gaps]

    def cost(j, i):  # 越接近估算位置、停顿越长越好
        return 4 * abs(mids[i] - expect[j]) / span - min(gaps[i][1] - gaps[i][0], 60) / 60

    inf = float("inf")
    dp = [[inf] * n for _ in range(k)]
    back = [[-1] * n for _ in range(k)]
    for i in range(n):
        dp[0][i] = cost(0, i)
    for j in range(1, k):
        best, arg = inf, -1
        for i in range(n):
            if i > 0 and dp[j - 1][i - 1] < best:
                best, arg = dp[j - 1][i - 1], i - 1
            if arg >= 0:
                dp[j][i] = best + cost(j, i)
                back[j][i] = arg
    i = int(np.argmin(dp[k - 1]))
    if dp[k - 1][i] == inf:
        return None
    cuts = [i]
    for j in range(k - 1, 0, -1):
        i = back[j][i]
        cuts.append(i)
    cuts.reverse()
    edges = [first] + [x for c in cuts for x in gaps[c]] + [last]
    pieces = [(edges[2 * i], edges[2 * i + 1]) for i in range(k + 1)]
    for (s, e), w in zip(pieces, weights):  # 每句时长和字数比例差太多，说明切错了
        ratio = ((e - s) / span) / (w / total)
        if not 0.4 <= ratio <= 2.2:
            return None
    return pieces


def cut_piece(audio, sr, start_frame, end_frame, lo, hi):
    hop = int(sr * FRAME)
    s = max(lo, start_frame * hop - int(0.03 * sr))
    e = min(hi, end_frame * hop + int(0.08 * sr))
    return audio[s:e].copy()


def squeeze_pauses(audio, sr, max_pause=0.45, keep=0.3):
    """句中超过 max_pause 的停顿缩到 keep 秒（模型偶尔会停得太久）。"""
    mask = voiced(audio, sr)
    out, pos = [], 0
    for s, e in runs(mask, False):
        if s == 0 or e >= len(mask) or (e - s) * FRAME <= max_pause:
            continue
        out.append(audio[pos: int((s * FRAME + keep / 2) * sr)])
        pos = int((e * FRAME - keep / 2) * sr)
    out.append(audio[pos:])
    return np.concatenate(out)


def fade(audio, sr, ms=8):
    n = min(len(audio) // 2, int(sr * ms / 1000))
    if n > 0:
        ramp = np.linspace(0, 1, n)
        audio[:n] *= ramp
        audio[-n:] *= ramp[::-1]
    return audio


def speech_rms_db(audio, sr):
    mask = voiced(audio, sr)
    hop = int(sr * FRAME)
    frames = [audio[i * hop:(i + 1) * hop] for i in np.where(mask)[0]]
    if not frames:
        return -90.0
    return 20 * np.log10(np.sqrt(np.mean(np.concatenate(frames) ** 2)) + 1e-12)


# ---------- ffmpeg ----------

def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("需要 ffmpeg：pip install imageio-ffmpeg（或把 ffmpeg 加进 PATH）")


def run_ffmpeg(*args, capture=False):
    cmd = [ffmpeg_exe(), "-hide_banner", "-y", *map(str, args)]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError("ffmpeg 失败：" + res.stderr.decode("utf-8", "replace")[-800:])
    return res.stderr.decode("utf-8", "replace") if capture else None


def decode_audio(path, sr, channels=2):
    res = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", str(path), "-vn", "-f", "f32le",
                          "-ac", str(channels), "-ar", str(sr), "-"], capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"读不出 {path} 的声音：" + res.stderr.decode("utf-8", "replace")[-400:])
    return np.frombuffer(res.stdout, dtype=np.float32).reshape(-1, channels).astype(np.float64)


def atempo(audio, sr, factor):
    if abs(factor - 1) < 0.005:
        return audio
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
        sf.write(str(src), audio, sr, subtype="FLOAT")
        run_ffmpeg("-i", src, "-filter:a", f"atempo={factor:.4f}", "-ar", sr, dst)
        out, _ = sf.read(str(dst), dtype="float64")
    return out


# ---------- 发音核对（可选） ----------

def make_asr(enabled):
    if not enabled:
        return None
    try:
        from funasr import AutoModel
        from scipy.signal import resample_poly
        model = AutoModel(model="paraformer-zh", disable_update=True)
    except Exception as exc:  # 没装或下载失败都不影响配音
        print(f"（语音识别核对不可用，跳过：{exc}）")
        return None

    def transcribe(audio, sr):
        audio16 = resample_poly(audio, 16000, sr).astype(np.float32)
        result = model.generate(input=audio16)
        return result[0]["text"] if result else ""
    return transcribe


def char_error_rate(ref, hyp):
    """只比较汉字：多读、漏读、读错字都会体现出来（多音字读错识别不出来，要靠耳朵）。"""
    r, h = re.findall(r"[一-鿿]", ref), re.findall(r"[一-鿿]", hyp)
    if not r:
        return 0.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)


# ---------- 选条、切句、落位 ----------

def flat_lines(data):
    lines = [line for g in data["groups"] for line in g["lines"]]
    for line, nxt in zip(lines, lines[1:] + [None]):
        line["next_start"] = nxt["start"] if nxt else data["duration"]
    return lines


def prepare_take(audio, sr, group, max_speed):
    """把一条生成结果切成句、去掉过长停顿，算出需要的加速倍数。"""
    texts = [line["synth"] for line in group["lines"]]
    split = split_lines(audio, sr, texts)
    hop = int(sr * FRAME)
    if split:
        bounds = [0] + [int((split[i][1] + split[i + 1][0]) / 2 * hop) for i in range(len(split) - 1)] + [len(audio)]
        pieces = [cut_piece(audio, sr, s, e, bounds[i], bounds[i + 1]) for i, (s, e) in enumerate(split)]
        targets = group["lines"]
    else:  # 切不开就整组放在第一句的位置
        on = np.where(voiced(audio, sr))[0]
        pieces = [cut_piece(audio, sr, int(on[0]), int(on[-1]) + 1, 0, len(audio))] if len(on) else []
        first, last = group["lines"][0], group["lines"][-1]
        targets = [dict(first, end=last["end"], next_start=last["next_start"])] if pieces else []
    pieces = [squeeze_pauses(p, sr) for p in pieces]
    allowed = [min(t["next_start"] - t["start"] - 0.06, t["end"] - t["start"] + 0.35) for t in targets]
    need = max((len(p) / sr / a for p, a in zip(pieces, allowed)), default=9.9)
    return {"pieces": pieces, "targets": targets, "split_ok": bool(split), "need": need,
            "factor": min(max(need, 1.0), max_speed)}


def take_penalty(prep, cer, max_speed):
    need = prep["need"]
    penalty = (need - 1) * 6 if need > 1 else (1 - need) * 0.4
    if need > max_speed:
        penalty += 4
    if not prep["split_ok"]:
        penalty += 3
    if cer is not None:
        penalty += 15 * max(0.0, cer - 0.02)
    return penalty


def choose_takes(data, takes_dir, max_speed, asr):
    entries = read_manifest(takes_dir)
    ratings = read_ratings(takes_dir)
    cache_path = takes_dir / "asr.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    chosen = {}
    for group in data["groups"]:
        candidates = []
        for e in entries:
            if e.get("group") != group["id"] or e.get("text") != group["text"]:
                continue
            audio, sr = sf.read(str(takes_dir / e["file"]), dtype="float64")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            cer = None
            if asr:
                if e["file"] not in cache:
                    hyp = asr(audio, sr)
                    cache[e["file"]] = {"text": hyp, "cer": round(char_error_rate(" ".join(l["text"] for l in group["lines"]), hyp), 3)}
                cer = cache[e["file"]]["cer"]
            elif e["file"] in cache:
                cer = cache[e["file"]]["cer"]
            prep = prepare_take(audio, sr, group, max_speed)
            score = (ratings.get(e["file"]) or {}).get("score") or 0
            candidates.append(dict(entry=e, sr=sr, prep=prep, cer=cer, score=score,
                                   penalty=take_penalty(prep, cer, max_speed)))
        if not candidates:
            raise SystemExit(f"组 {group['id']} 还没有生成结果，先运行 render")
        candidates.sort(key=lambda c: (-c["score"], c["penalty"]))
        chosen[group["id"]] = candidates[0]
    if asr:
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return chosen


def assemble(data, chosen, sr, max_speed):
    track = np.zeros(int(round(data["duration"] * sr)))
    placed, warnings, cursor = [], [], 0.0
    for group in data["groups"]:
        c = chosen[group["id"]]
        prep = c["prep"]
        if c["sr"] != sr:
            raise SystemExit(f"组 {group['id']} 的采样率是 {c['sr']}，和其它组的 {sr} 不一致")
        if not prep["pieces"]:
            warnings.append(f"组 {group['id']}：选中的那条几乎是静音，已跳过，请重新生成这一组")
            continue
        gain = 10 ** ((-20 - speech_rms_db(np.concatenate(prep["pieces"]), sr)) / 20)
        if prep["need"] > max_speed:
            warnings.append(f"组 {group['id']}：需要加速 {prep['need']:.2f} 倍才放得下，已按上限 {max_speed} 倍处理，后面的句子会顺延")
        if not prep["split_ok"]:
            warnings.append(f"组 {group['id']}：没能按句切开，整组从第一句的时间开始播放")
        if c["cer"] is not None and c["cer"] > 0.05:
            warnings.append(f"组 {group['id']}：识别出的文字和稿子差 {c['cer']:.0%}，可能有漏读或读错，务必听一下")
        for piece, target in zip(prep["pieces"], prep["targets"]):
            piece = fade(atempo(piece * gain, sr, prep["factor"]), sr)
            peak = np.abs(piece).max()
            if peak > 0.89:
                piece *= 0.89 / peak
            start = max(target["start"], cursor + 0.06 if cursor else 0.0)
            i = int(round(start * sr))
            piece = piece[: max(0, len(track) - i)]
            track[i: i + len(piece)] += piece
            end = start + len(piece) / sr
            cursor = end
            late = start - target["start"]
            if late > 0.25:
                warnings.append(f"{target['shot']}「{target['text']}」比字幕晚 {late:.2f} 秒开始")
            placed.append(dict(shot=target["shot"], text=target["text"], window=[target["start"], target["end"]],
                               start=round(start, 2), end=round(end, 2), group=group["id"],
                               take=c["entry"]["take"], file=c["entry"]["file"], speed=round(prep["factor"], 3),
                               cer=c["cer"], split_ok=prep["split_ok"]))
    return track, placed, warnings


# ---------- 混音与合成 ----------

def duck_envelope(n, sr, intervals, duck_db, pre=0.2, post=0.35, ramp=0.25):
    target = np.zeros(n)
    for s, e in intervals:
        target[max(0, int((s - pre) * sr)): min(n, int((e + post) * sr))] = -duck_db
    k = max(1, int(ramp * sr))
    kernel = np.hanning(k)
    kernel /= kernel.sum()
    smooth = np.convolve(target, kernel, mode="same")
    return 10 ** (smooth / 20)


def loudnorm(src, dst, lufs, sr):
    stats = run_ffmpeg("-i", src, "-af", f"loudnorm=I={lufs}:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-",
                       capture=True)
    m = json.loads(stats[stats.rindex("{"): stats.rindex("}") + 1])
    run_ffmpeg("-i", src, "-af",
               f"loudnorm=I={lufs}:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
               f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true",
               "-ar", sr, "-c:a", "pcm_s24le", dst)


def mix_and_mux(args, track, sr, placed, out_dir):
    n = len(track)
    if args.music:
        bed = decode_audio(args.music, sr)
        if args.sfx:
            sfx = decode_audio(args.sfx, sr)
            m = min(len(bed), len(sfx))
            bed = bed[:m] + sfx[:m]
        source = "music.wav + sfx.wav"
    else:
        bed = decode_audio(args.video, sr)
        source = "视频原有音轨"
    bed = np.pad(bed, ((0, max(0, n - len(bed))), (0, 0)))[:n]
    env = duck_envelope(n, sr, [(p["start"], p["end"]) for p in placed], args.duck_db)
    ducked = bed * env[:, None]

    speech = np.zeros(n, dtype=bool)
    for p in placed:
        speech[int(p["start"] * sr): int(p["end"] * sr)] = True
    bed_db = 20 * np.log10(np.sqrt(np.mean(ducked[speech] ** 2)) + 1e-12) if speech.any() else -30
    voice_gain = 10 ** ((bed_db + args.voice_over_bed - speech_rms_db(track, sr)) / 20)
    mix = ducked + (track * voice_gain)[:, None]

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "mix_raw.wav"
        sf.write(str(raw), mix, sr, subtype="FLOAT")
        loudnorm(raw, out_dir / "mix.wav", args.lufs, sr)

    video_out = out_dir / args.output_name
    duration = n / sr
    probe = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(args.video)], capture_output=True)
    video_kbps = re.search(r"Video:.*?(\d+) kb/s", probe.stderr.decode("utf-8", "replace"))
    video_bytes = int(video_kbps.group(1)) * 1000 / 8 * duration if video_kbps else Path(args.video).stat().st_size
    audio_kbps = int(min(192, (args.size_limit * 0.975 - video_bytes) * 8 / duration / 1000))
    for kbps in [audio_kbps, audio_kbps - 32, 96]:
        kbps = max(96, kbps)
        run_ffmpeg("-i", args.video, "-i", out_dir / "mix.wav", "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                   "-c:a", "aac", "-b:a", f"{kbps}k", "-ar", sr, "-movflags", "+faststart", "-shortest", video_out)
        size = video_out.stat().st_size
        if size <= args.size_limit:
            break
    return {"bed": source, "duck_db": args.duck_db, "voice_over_bed_db": args.voice_over_bed, "lufs": args.lufs,
            "audio_kbps": kbps, "video_file": str(video_out), "video_bytes": size,
            "under_limit": size <= args.size_limit}


def write_outputs(args, data, track, sr, placed, warnings, out_dir):
    final = out_dir / "final"
    (final / "segments").mkdir(parents=True, exist_ok=True)
    with open(final / "人声轨_48k24bit.wav", "wb") as f:
        sf.write(f, track, sr, format="WAV", subtype="PCM_24")
    for name, (s, e) in data["segments"].items():
        with open(final / "segments" / f"{name}.wav", "wb") as f:
            sf.write(f, track[int(s * sr): int(e * sr)], sr, format="WAV", subtype="PCM_24")
    mix_info = mix_and_mux(args, track, sr, placed, final) if args.video else None

    report = {"ref": args.ref, "mode": args.mode, "lines": placed, "warnings": warnings, "mix": mix_info}
    (final / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    rows = ["| 镜头 | 台词 | 字幕时间 | 实际 | 加速 | 用的哪条 | 识别差异 |", "|---|---|---|---|---|---|---|"]
    for p in placed:
        cer = "" if p["cer"] is None else f"{p['cer']:.0%}"
        rows.append(f"| {p['shot']} | {p['text']} | {p['window'][0]:.2f}–{p['window'][1]:.2f} | "
                    f"{p['start']:.2f}–{p['end']:.2f} | {p['speed']:g} | {p['group']} 第{p['take']}条 | {cer} |")
    md = [f"# 配音报告（{args.ref} · {MODE_LABELS[args.mode]}）", ""]
    md += ["## 需要注意", ""] + ([f"- {w}" for w in warnings] or ["- 没有自动发现的问题，仍需人工试听。"])
    md += ["", "## 试听时重点听这些读音", ""] + [f"- {w}" for w in data["check_words"]]
    if mix_info:
        md += ["", "## 混音与视频", "",
               f"- 底乐来源：{mix_info['bed']}；说话时底乐压低 {mix_info['duck_db']} dB，人声比压低后的底乐响 {mix_info['voice_over_bed_db']} dB",
               f"- 整体响度 {mix_info['lufs']} LUFS；音频 AAC {mix_info['audio_kbps']} kbps",
               f"- 视频：{mix_info['video_file']}，{mix_info['video_bytes'] / 1e6:.2f} MB"
               + ("" if mix_info["under_limit"] else f"（⚠ 超过 {args.size_limit / 1e6:.0f} MB 上限）")]
    md += ["", "## 逐句落位", ""] + rows
    (final / "配音报告.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n人声轨：{final / '人声轨_48k24bit.wav'}")
    if mix_info:
        print(f"成片：{mix_info['video_file']}（{mix_info['video_bytes'] / 1e6:.2f} MB）")
    print(f"报告：{final / '配音报告.md'}")
    for w in warnings:
        print("⚠ " + w)


# ---------- 命令 ----------

def add_common(parser):
    parser.add_argument("--script", default=str(SCRIPT), help="配音数据文件（默认 dub/xiaonian_v10.json）")
    parser.add_argument("--model-path", help="VoxCPM2 本地路径")
    parser.add_argument("--compile", action="store_true", help="尝试 torch.compile 加速")
    parser.add_argument("--cfg", type=float, default=2.0, help="VoxCPM2 引导强度（默认 2.0）")
    parser.add_argument("--steps", type=int, default=10, help="VoxCPM2 推理步数（默认 10）")


def add_assemble(parser):
    parser.add_argument("--ref", required=True, help="选定的参考音色（候选 id，或 wav 路径配合 --ref-text）")
    parser.add_argument("--ref-text", help="参考 wav 说的内容（--ref 给的是路径时需要）")
    parser.add_argument("--mode", choices=sorted(MODE_LABELS), default="style",
                        help="style：克隆音色 + 每段语气控制（默认）；clone：高保真克隆，语气跟着参考音频走")
    parser.add_argument("--video", help="成片视频（给了就会混音并输出配音版视频）")
    parser.add_argument("--music", help="单独的 BGM 分轨（有的话比用视频原音轨更好）")
    parser.add_argument("--sfx", help="单独的音效分轨，和 --music 一起用")
    parser.add_argument("--output-name", default="晓念V10_配音版.mp4", help="输出视频文件名")
    parser.add_argument("--max-speed", type=float, default=1.10, help="放不下时最多加速到几倍（默认 1.10）")
    parser.add_argument("--duck-db", type=float, default=8.0, help="说话时底乐压低多少 dB（默认 8，稿子建议 6–9）")
    parser.add_argument("--voice-over-bed", type=float, default=10.0, help="人声比压低后的底乐响多少 dB（默认 10）")
    parser.add_argument("--lufs", type=float, default=-16.0, help="成片整体响度（默认 -16 LUFS）")
    parser.add_argument("--size-limit", type=int, default=30_000_000, help="成片大小上限，字节（默认 30MB）")
    parser.add_argument("--no-asr", action="store_true", help="不用语音识别核对漏读、错读")


def out_dir_for(args):
    return OUT_ROOT / f"{Path(args.ref).stem}_{args.mode}"


def cmd_test(args):
    data = load_script(args.script)
    groups = pick_groups(data, args.groups)
    refs = [r.strip() for r in args.refs.split(",") if r.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    gen = {"cfg": args.cfg, "steps": args.steps}
    jobs = [j for ref in refs for mode in modes for j in take_jobs(data, groups, ref, mode, [1], gen, label_by_ref=True)]
    out = OUT_ROOT / "test"
    run_jobs(jobs, out, lambda: VoxCPM2Backend(args.model_path, compile=args.compile), dry_run=args.dry_run)
    if not args.dry_run:
        print(f"试配完成。用试音台盲听 {out}，评完看统计里的「方向」（= 音色）和「精调变体」（= 方式）。")


def cmd_render(args):
    data = load_script(args.script)
    flat_lines(data)
    out = out_dir_for(args)
    takes_dir = out / "takes"
    gen = {"cfg": args.cfg, "steps": args.steps}
    jobs = take_jobs(data, pick_groups(data, args.only), args.ref, args.mode, range(1, args.takes + 1), gen)
    run_jobs(jobs, takes_dir, lambda: VoxCPM2Backend(args.model_path, compile=args.compile), dry_run=args.dry_run)
    if not args.dry_run:
        do_assemble(args, data, takes_dir, out)


def do_assemble(args, data, takes_dir, out):
    print("\n正在挑选每组最合适的一条、切句、落位……")
    chosen = choose_takes(data, takes_dir, args.max_speed, make_asr(not args.no_asr))
    sr = next(iter(chosen.values()))["sr"]
    track, placed, warnings = assemble(data, chosen, sr, args.max_speed)
    write_outputs(args, data, track, sr, placed, warnings, out)


def cmd_assemble(args):
    data = load_script(args.script)
    flat_lines(data)
    out = out_dir_for(args)
    if not (out / "takes").exists():
        raise SystemExit(f"{out / 'takes'} 不存在，先运行 render")
    do_assemble(args, data, out / "takes", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("test", help="试配：几个候选音色各配几组，盲听选音色")
    add_common(p)
    p.add_argument("--refs", required=True, help="候选音色 id，逗号分隔")
    p.add_argument("--modes", default="style", help="比较哪几种方式：style、clone，逗号分隔（默认只试 style）")
    p.add_argument("--groups", default="05-06,07,14,23a", help="试配哪几组（默认介绍、提问、研究、技术各一组）")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_test)

    p = sub.add_parser("render", help="正式配音：生成、挑选、落位、混音、合成视频")
    add_common(p)
    add_assemble(p)
    p.add_argument("--takes", type=int, default=3, help="每组生成几条备选（默认 3）")
    p.add_argument("--only", help="只生成某几组，如 05-06,14（其它组用已有结果）")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("assemble", help="不再生成，按现有结果（和试音台评分）重新挑选、落位、混音")
    add_common(p)
    add_assemble(p)
    p.set_defaults(func=cmd_assemble)

    args = parser.parse_args()
    if getattr(args, "video", None) and not Path(args.video).exists():
        raise SystemExit(f"找不到视频：{args.video}")
    args.func(args)


if __name__ == "__main__":
    main()
