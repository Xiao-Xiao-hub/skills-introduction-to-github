"""Mix the accepted C3 narration with the V10 music/SFX stems and remux it.

This deliberately does not synthesize, retime, gate, or otherwise edit the
narration.  It consumes the 48 kHz/24-bit aligned track produced upstream and
uses a single whole-track voice gain.  Music and SFX are decoded only to apply
their separate, time-aligned ducking envelopes; their source files are never
written to.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dub  # noqa: E402

SR = 48_000
OUT = ROOT / "dub" / "out" / "c3-video-2026-10-22"
ALIGNED = OUT / "aligned"
REVIEW = Path(r"D:\XiaoNian-Video\review\2026-10-21-v10")
VIDEO = REVIEW / "晓念_V10_完整片_无旁白.mp4"
MUSIC = REVIEW / "audio" / "music.wav"
SFX = REVIEW / "audio" / "sfx.wav"
FINAL = Path(r"D:\XiaoNian-Video\review\2026-10-22-v10-c3\晓念_V10_C3配音_完整片.mp4")
EXPECTED_SOURCE_VIDEO_SHA256 = "5e201129a909b74db069355cd734b0c1356a30ba3c46ec7948bc4bf0187d7fca"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rms_db(audio: np.ndarray, mask: np.ndarray | None = None) -> float:
    values = audio if mask is None else audio[mask]
    if values.size == 0:
        return -120.0
    return float(20 * np.log10(np.sqrt(np.mean(np.square(values, dtype=np.float64))) + 1e-12))


def read_alignment(path: Path, samples: int, waveform_sha256: str) -> list[dict[str, Any]]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"alignment.json 无法读取：{exc}") from exc
    if not isinstance(rows, dict) or not isinstance(rows.get("rows"), list) or len(rows["rows"]) != 55:
        raise SystemExit("alignment.json 必须是含恰好 55 条 rows 的对象")
    if rows.get("waveform_sha256") != waveform_sha256:
        raise SystemExit("alignment.json 的 waveform_sha256 与输入人声轨不一致")
    result = rows["rows"]
    previous_end = 0.0
    duration = samples / SR
    one_sample = 1.0 / SR
    for expected, row in enumerate(result, start=1):
        if not isinstance(row, dict):
            raise SystemExit(f"alignment 第 {expected} 条不是对象")
        required = ("index", "group", "text", "window", "start", "end")
        missing = [key for key in required if key not in row]
        if missing or row["index"] != expected or not isinstance(row["group"], str) or not isinstance(row["text"], str):
            raise SystemExit(f"alignment 第 {expected} 条字段或 index 不正确")
        window = row["window"]
        if not isinstance(window, list) or len(window) != 2:
            raise SystemExit(f"alignment 第 {expected} 条 window 不正确")
        try:
            start, end = float(row["start"]), float(row["end"])
            left, right = float(window[0]), float(window[1])
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"alignment 第 {expected} 条时间不是数值") from exc
        if not (0 <= left and start <= end and start >= left - one_sample and end <= right + one_sample and
                end <= duration + one_sample and start >= previous_end):
            raise SystemExit(f"alignment 第 {expected} 条时间越界或逆序")
        previous_end = end
    return result


def read_track(path: Path) -> np.ndarray:
    info = sf.info(str(path))
    if info.samplerate != SR or info.channels != 1 or info.subtype != "PCM_24":
        raise SystemExit(f"人声必须为 48kHz/单声道/PCM_24，实际为 {info.samplerate}/{info.channels}/{info.subtype}")
    audio, _ = sf.read(str(path), dtype="float32", always_2d=True)
    return np.ascontiguousarray(audio[:, 0])


def read_stem(path: Path, target_samples: int) -> np.ndarray:
    """Use the existing decoder, but keep the working copy compact (float32)."""
    decoded = dub.decode_audio(path, SR, channels=2).astype(np.float32, copy=False)
    if len(decoded) < target_samples:
        decoded = np.pad(decoded, ((0, target_samples - len(decoded)), (0, 0)))
    return np.ascontiguousarray(decoded[:target_samples])


def duck_depth_db(samples: int, intervals: list[tuple[float, float]], duck_db: float,
                  lookahead: float = 0.20, hold: float = 0.35, ramp: float = 0.25) -> np.ndarray:
    """A 100 Hz shaped envelope, lightly Hann-smoothed with fftconvolve, then expanded.

    The low-rate construction avoids a full-rate convolution over the 178-second
    stems.  Each cue ramps in during the 200 ms lookahead, holds through the
    requested post-speech hold, and ramps out over 250 ms.
    """
    rate = 100
    n = math.ceil(samples * rate / SR) + 2
    depth = np.zeros(n, dtype=np.float32)
    for start, end in intervals:
        a = max(0, int(math.floor((start - lookahead) * rate)))
        b = min(n, int(math.ceil(start * rate)))
        c = min(n, int(math.ceil((end + hold) * rate)))
        d = min(n, int(math.ceil((end + hold + ramp) * rate)))
        if b > a:
            depth[a:b] = np.maximum(depth[a:b], np.linspace(0, duck_db, b - a, endpoint=False, dtype=np.float32))
        if c > b:
            depth[b:c] = np.maximum(depth[b:c], duck_db)
        if d > c:
            depth[c:d] = np.maximum(depth[c:d], np.linspace(duck_db, 0, d - c, endpoint=False, dtype=np.float32))
    # Five 10 ms bins rounds envelope corners without moving its intended timing.
    kernel = np.hanning(5).astype(np.float32)
    kernel /= kernel.sum()
    depth = fftconvolve(depth, kernel, mode="same").astype(np.float32, copy=False)
    # Interpolation prevents 10 ms staircase zippering while retaining the
    # compact low-rate construction.
    low_times = np.arange(n, dtype=np.float64) / rate
    sample_times = np.arange(samples, dtype=np.float64) / SR
    return np.interp(sample_times, low_times, depth).astype(np.float32, copy=False)


def loudnorm_stats(path: Path) -> dict[str, str]:
    output = dub.run_ffmpeg("-i", path, "-af", "loudnorm=I=-16:TP=-2.0:LRA=11:print_format=json",
                            "-f", "null", "-", capture=True)
    try:
        return json.loads(output[output.rindex("{"):output.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("ffmpeg loudnorm 未返回可解析统计") from exc


def loudnorm_two_pass(raw: Path, destination: Path) -> dict[str, str]:
    first = loudnorm_stats(raw)
    filt = ("loudnorm=I=-16:TP=-2.0:LRA=11:"
            f"measured_I={first['input_i']}:measured_TP={first['input_tp']}:"
            f"measured_LRA={first['input_lra']}:measured_thresh={first['input_thresh']}:"
            f"offset={first['target_offset']}:linear=true:print_format=json")
    dub.run_ffmpeg("-i", raw, "-af", filt, "-ar", SR, "-ac", "2", "-c:a", "pcm_s24le", destination)
    return {"first_pass": first, "normalized_wav": loudnorm_stats(destination)}


def ffprobe(path: Path) -> dict[str, Any]:
    exe = shutil.which("ffprobe")
    if not exe:
        raise SystemExit("需要 ffprobe（与已安装 ffmpeg 同一发行版）")
    result = subprocess.run([exe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                            capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[-800:])
    return json.loads(result.stdout)


def video_hash(path: Path) -> str:
    result = subprocess.run([dub.ffmpeg_exe(), "-v", "error", "-i", str(path), "-map", "0:v:0", "-c", "copy", "-f", "hash", "-hash", "sha256", "-"],
                            capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[-800:])
    line = result.stdout.decode("utf-8", "replace").strip()
    if not line.startswith("SHA256="):
        raise RuntimeError(f"无法取得视频流 SHA256：{line}")
    return line.partition("=")[2]


def validate_final(path: Path, source_video_hash: str) -> dict[str, Any]:
    probe = ffprobe(path)
    streams = probe["streams"]
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not video or not audio:
        raise RuntimeError("输出缺少视频或音频流")
    frame_rate = video.get("avg_frame_rate")
    frames = int(video.get("nb_frames", 0) or 0)
    duration = float(probe["format"]["duration"])
    checks = {"1080p": (video.get("width"), video.get("height")) == (1920, 1080),
              "60fps": frame_rate == "60/1", "10680_frames": frames == 10680,
              "178_seconds": abs(duration - 178.0) <= 0.10,
              "aac_stereo_48k": audio.get("codec_name") == "aac" and int(audio.get("sample_rate", 0)) == SR and int(audio.get("channels", 0)) == 2,
              "aac_bitrate_near_256k": 230_000 <= int(audio.get("bit_rate", 0) or 0) <= 280_000,
              "video_stream_hash_matches": video_hash(path) == source_video_hash}
    # Full decode is the final corruption check, rather than relying on a muxer exit code.
    dub.run_ffmpeg("-xerror", "-v", "error", "-i", path, "-map", "0", "-f", "null", "-")
    if not all(checks.values()):
        raise RuntimeError("成片检查失败：" + json.dumps(checks, ensure_ascii=False))
    return {"checks": checks, "duration_seconds": duration, "video": video, "audio": audio}


def main() -> int:
    parser = argparse.ArgumentParser(description="C3 narration/music/SFX mixer; no synthesis or narration edits")
    parser.add_argument("--track", type=Path, default=ALIGNED / "人声轨_48k24bit.wav")
    parser.add_argument("--alignment", type=Path, default=ALIGNED / "alignment.json")
    parser.add_argument("--video", type=Path, default=VIDEO)
    parser.add_argument("--music", type=Path, default=MUSIC)
    parser.add_argument("--sfx", type=Path, default=SFX)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--final", type=Path, default=FINAL)
    args = parser.parse_args()
    track_path, alignment_path = args.track.resolve(), args.alignment.resolve()
    out, final = args.out.resolve(), args.final.resolve()
    sources = [track_path, alignment_path, args.video.resolve(), args.music.resolve(), args.sfx.resolve()]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise SystemExit("缺少输入：\n" + "\n".join(missing))
    source_snapshots = {str(path): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in sources}
    if args.video.resolve() == VIDEO.resolve() and sha256(args.video) != EXPECTED_SOURCE_VIDEO_SHA256:
        raise RuntimeError("指定原片文件 SHA256 与已核验值不符")
    out.mkdir(parents=True, exist_ok=True)
    (out / "aligned").mkdir(exist_ok=True)
    mix_path, report_path, metadata_path = out / "mix.wav", out / "mix-report.json", out / "aligned" / "mix-metadata.json"
    occupied = [path for path in (mix_path, report_path, metadata_path, final) if path.exists()]
    if occupied:
        raise SystemExit("为避免覆盖现有交付，以下输出已存在：\n" + "\n".join(map(str, occupied)))

    voice = read_track(track_path)
    track_hash = sha256(track_path)
    rows = read_alignment(alignment_path, len(voice), track_hash)
    intervals = [(float(row["start"]), float(row["end"])) for row in rows]
    mask = np.zeros(len(voice), dtype=bool)
    for start, end in intervals:
        mask[max(0, round(start * SR)):min(len(mask), round(end * SR))] = True
    music, sfx = read_stem(args.music, len(voice)), read_stem(args.sfx, len(voice))
    depth = duck_depth_db(len(voice), intervals, duck_db=8.0)
    music_gain = np.power(10.0, -depth / 20.0, dtype=np.float32)
    sfx_gain = np.power(10.0, -(depth * (2.5 / 8.0)) / 20.0, dtype=np.float32)
    # The short brand name was masked in the first encoded mix.  Give all three
    # closing cues another 6 dB of smooth bed reduction, with the same ramps.
    closing_intervals = [(float(row['start']), float(row['end'])) for row in rows if row['index'] >= 53]
    closing_depth = duck_depth_db(len(voice), closing_intervals, duck_db=6.0)
    closing_bed_gain = np.power(10.0, -closing_depth / 20.0, dtype=np.float32)
    music_gain *= closing_bed_gain
    sfx_gain *= closing_bed_gain
    music *= music_gain[:, None]
    sfx *= sfx_gain[:, None]
    bed = music + sfx
    voice_before = rms_db(voice, mask)
    bed_active = rms_db(bed, mask)
    voice_gain_db = bed_active + 10.0 - voice_before
    voice_gain = np.float32(10 ** (voice_gain_db / 20.0))
    mix = bed.copy()
    mix += (voice * voice_gain)[:, None]

    with tempfile.TemporaryDirectory(prefix="c3-video-mix-") as temp:
        raw = Path(temp) / "mix-pre-loudnorm.wav"
        sf.write(str(raw), mix, SR, subtype="FLOAT")
        loudness = loudnorm_two_pass(raw, mix_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    source_video_hash = video_hash(args.video)
    dub.run_ffmpeg("-i", args.video, "-i", mix_path, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                   "-c:a", "aac", "-b:a", "256k", "-ar", SR, "-ac", "2", "-movflags", "+faststart", "-shortest", final)
    validation = validate_final(final, source_video_hash)
    final_loudness = loudnorm_stats(final)
    actual_i, actual_tp = float(final_loudness["input_i"]), float(final_loudness["input_tp"])
    # Normalize PCM to -2 dBTP to reserve headroom for AAC reconstruction peaks.
    loudness_checks = {"integrated_lufs_within_0_5": -16.5 <= actual_i <= -15.5,
                       "true_peak_not_over_minus_1_5_dbtp": actual_tp <= -1.5}
    if not all(loudness_checks.values()):
        raise RuntimeError("最终 AAC 响度/真峰值不在允许范围：" + json.dumps({"I": actual_i, "TP": actual_tp, **loudness_checks}))
    cue_metrics = []
    for row in rows:
        start, end = round(float(row["start"]) * SR), round(float(row["end"]) * SR)
        cue_metrics.append({"index": row["index"], "group": row["group"], "text": row["text"],
                            "window": row["window"], "start": row["start"], "end": row["end"],
                            "voice_rms_dbfs": round(rms_db(voice[start:end] * voice_gain), 3),
                            "ducked_bed_rms_dbfs": round(rms_db(bed[start:end]), 3)})
    assert all(sha256(path) == source_snapshots[str(path)]['sha256'] for path in sources), 'Source changed during mix'
    report = {"status": "passed", "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
              "sources": source_snapshots, "source_video_packet_sha256": source_video_hash,
              "outputs": {"mix_wav": str(mix_path), "final_mp4": str(final)},
              "parameters": {"sample_rate": SR, "voice_gain_db": round(voice_gain_db, 6), "voice_over_ducked_combined_bed_target_db": 10.0,
                             "music_duck_db": 8.0, "sfx_duck_db": 2.5, "closing_extra_bed_duck_db": 6.0, "lookahead_seconds": 0.20, "hold_seconds": 0.35,
                             "ramp_seconds": 0.25, "envelope_rate_hz": 100, "envelope_smoother": "scipy.signal.fftconvolve Hann(5)",
                             "normalization": "ffmpeg loudnorm two-pass I=-16 TP=-2.0 LRA=11; final AAC ceiling -1.5 dBTP", "narration_processing": "one whole-track gain; no gate or pitch edit"},
              "active_speech_rms_dbfs": {"voice_before_gain": round(voice_before, 3), "voice_after_gain": round(rms_db(voice * voice_gain, mask), 3),
                                            "ducked_music_plus_sfx": round(bed_active, 3), "voice_minus_bed": round(rms_db(voice * voice_gain, mask) - bed_active, 3)},
              "per_cue": cue_metrics, "loudness": {"wav": loudness, "final_aac_measured": final_loudness,
                                                      "final_aac_tolerance_checks": loudness_checks},
              "validation": validation, "original_stems_unchanged": True}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps({"track": str(track_path), "alignment": str(alignment_path), "cue_count": len(rows),
                                          "track_sha256": track_hash, "alignment_sha256": sha256(alignment_path)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mix": str(mix_path), "report": str(report_path), "final": str(final)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
