"""C3 一次性自然拼接：整组保留停顿，只做必要的整体变速和受控增益。

默认输入：dub/out/c3-expression-2026-10-22/selection.json 和 takes/
默认输出：dub/out/c3-expression-2026-10-22/natural-final/

这个助手不切句、不重排逐条字幕，也不运行生成模型或 ASR。它只读取原始
``dub.load_script(dub.SCRIPT)`` 的时间窗，把选中的整组音频落到该组首条字幕
的起点。报告中的 cue 窗口仅供审计，不代表 55 条 cue 已逐条精确对齐。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dub  # noqa: E402


SOURCE = ROOT / "dub" / "out" / "c3-expression-2026-10-22"
DEFAULT_SELECTION = SOURCE / "selection.json"
DEFAULT_TAKES = SOURCE / "takes"
DEFAULT_OUTPUT = SOURCE / "natural-final"

TARGET_SR = 48_000
HEAD_KEEP_SECONDS = 0.060
TAIL_KEEP_SECONDS = 0.120
END_GRACE_SECONDS = 0.350
NEXT_GAP_SECONDS = 0.060
MAX_SPEED = 1.05
TARGET_MEDIAN_RMS_DB = -20.0
GROUP_RMS_TOLERANCE_DB = 2.0
PEAK_LIMIT = 0.89
MIN_REPORTED_PAUSE_SECONDS = 0.08
WAVE_NAME = "人声轨_48k24bit.wav"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_seconds(value: float) -> float:
    return round(float(value), 6)


def db_to_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)


def safe_relative_file(base: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise SystemExit("selection.json 中的 file 必须是非空相对路径")
    candidate = (base / relative).resolve()
    resolved_base = base.resolve()
    if candidate != resolved_base and resolved_base not in candidate.parents:
        raise SystemExit(f"选条路径越过 takes 目录：{relative}")
    return candidate


def load_selection(path: Path, takes_dir: Path, group_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"找不到选条文件：{path}")
    try:
        selected = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"selection.json 无法读取：{exc}") from exc
    if not isinstance(selected, dict):
        raise SystemExit("selection.json 顶层必须是按 group id 索引的对象")

    missing = [group_id for group_id in group_ids if group_id not in selected]
    extra = [group_id for group_id in selected if group_id not in group_ids]
    if missing or extra:
        raise SystemExit(f"selection.json 组不完整：缺少={missing}，多出={extra}")

    required = ("file", "seed", "reference_sha256", "gen")
    normalized: dict[str, dict[str, Any]] = {}
    for group_id in group_ids:
        entry = selected[group_id]
        if not isinstance(entry, dict):
            raise SystemExit(f"组 {group_id} 的选条记录必须是对象")
        absent = [key for key in required if key not in entry]
        if absent:
            raise SystemExit(f"组 {group_id} 缺少字段：{absent}")
        if not isinstance(entry["seed"], int) or isinstance(entry["seed"], bool):
            raise SystemExit(f"组 {group_id} 的 seed 必须是整数")
        ref_hash = entry["reference_sha256"]
        if not isinstance(ref_hash, str) or len(ref_hash) != 64:
            raise SystemExit(f"组 {group_id} 的 reference_sha256 格式不正确")
        if not isinstance(entry["gen"], dict):
            raise SystemExit(f"组 {group_id} 的 gen 必须是对象")
        wave_path = safe_relative_file(takes_dir, entry["file"])
        if not wave_path.is_file():
            raise SystemExit(f"组 {group_id} 找不到音频：{wave_path}")
        source_hash = sha256_file(wave_path)
        declared_hash = entry.get("sha256")
        if declared_hash is not None and declared_hash != source_hash:
            raise SystemExit(f"组 {group_id} 音频 SHA256 与 selection.json 不一致")
        normalized[group_id] = {**entry, "path": wave_path, "source_sha256": source_hash}
    return normalized


def read_mono(path: Path) -> tuple[np.ndarray, int, int, int]:
    info = sf.info(str(path))
    audio, sample_rate = sf.read(str(path), dtype="float64", always_2d=True)
    channels = int(audio.shape[1])
    mono = audio.mean(axis=1)
    if sample_rate != TARGET_SR:
        mono = dub.decode_audio(path, TARGET_SR, channels=1).reshape(-1)
    mono = np.asarray(mono, dtype=np.float64)
    if not len(mono) or not np.isfinite(mono).all() or float(np.max(np.abs(mono))) < 1e-7:
        raise SystemExit(f"音频为空、非有限值或近似静音：{path}")
    return mono, int(info.samplerate), channels, int(info.frames)


def trim_outer_silence(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, dict[str, float]]:
    """只裁首尾静音；不足的头尾保护量用零补齐，组内样本顺序保持不变。"""
    mask = dub.voiced(audio, sample_rate)
    active = np.flatnonzero(mask)
    if not len(active):
        raise SystemExit("静音检测没有找到有效语音")
    hop = int(sample_rate * dub.FRAME)
    voice_start = int(active[0]) * hop
    voice_end = min(len(audio), (int(active[-1]) + 1) * hop)
    head = int(round(HEAD_KEEP_SECONDS * sample_rate))
    tail = int(round(TAIL_KEEP_SECONDS * sample_rate))

    wanted_start = voice_start - head
    wanted_end = voice_end + tail
    source_start = max(0, wanted_start)
    source_end = min(len(audio), wanted_end)
    pad_head = max(0, -wanted_start)
    pad_tail = max(0, wanted_end - len(audio))
    trimmed = np.pad(audio[source_start:source_end], (pad_head, pad_tail))

    voice_start_in_trim = pad_head + voice_start - source_start
    voice_end_in_trim = pad_head + voice_end - source_start
    details = {
        "source_duration_seconds": round_seconds(len(audio) / sample_rate),
        "source_trim_start_seconds": round_seconds(source_start / sample_rate),
        "source_trim_end_seconds": round_seconds((len(audio) - source_end) / sample_rate),
        "added_head_padding_seconds": round_seconds(pad_head / sample_rate),
        "added_tail_padding_seconds": round_seconds(pad_tail / sample_rate),
        "preserved_head_seconds": round_seconds(voice_start_in_trim / sample_rate),
        "preserved_tail_seconds": round_seconds((len(trimmed) - voice_end_in_trim) / sample_rate),
        "trimmed_duration_seconds": round_seconds(len(trimmed) / sample_rate),
    }
    return trimmed, details


def natural_pauses(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    mask = dub.voiced(audio, sample_rate)
    hop = int(sample_rate * dub.FRAME)
    intervals = []
    for start, end in dub.runs(mask, False):
        duration = (end - start) * hop / sample_rate
        if start == 0 or end == len(mask) or duration < MIN_REPORTED_PAUSE_SECONDS:
            continue
        intervals.append({
            "start_seconds": round_seconds(start * hop / sample_rate),
            "end_seconds": round_seconds(end * hop / sample_rate),
            "duration_seconds": round_seconds(duration),
        })
    durations = [item["duration_seconds"] for item in intervals]
    return {
        "minimum_reported_seconds": MIN_REPORTED_PAUSE_SECONDS,
        "count": len(intervals),
        "total_seconds": round_seconds(sum(durations)),
        "max_seconds": round_seconds(max(durations, default=0.0)),
        "intervals": intervals,
    }


def cue_windows(group: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "shot": line["shot"],
            "text": line["text"],
            "start": round_seconds(line["start"]),
            "end": round_seconds(line["end"]),
        }
        for line in group["lines"]
    ]


def hard_window(data: dict[str, Any], index: int) -> dict[str, Any]:
    group = data["groups"][index]
    start = float(group["lines"][0]["start"])
    last_cue_end = float(group["lines"][-1]["end"])
    cue_limit = last_cue_end + END_GRACE_SECONDS
    if index + 1 < len(data["groups"]):
        next_start = float(data["groups"][index + 1]["lines"][0]["start"])
        next_limit = next_start - NEXT_GAP_SECONDS
    else:
        next_start = None
        next_limit = float(data["duration"])
    end = min(cue_limit, next_limit, float(data["duration"]))
    return {
        "start": round_seconds(start),
        "end": round_seconds(end),
        "available_seconds": round_seconds(end - start),
        "last_cue_end_plus_0_35": round_seconds(cue_limit),
        "next_group_start": None if next_start is None else round_seconds(next_start),
        "next_group_start_minus_0_06": None if next_start is None else round_seconds(next_limit),
    }


def tempo_to_fit(audio: np.ndarray, sample_rate: int, available: float) -> tuple[np.ndarray | None, dict[str, Any]]:
    head_samples = int(round(HEAD_KEEP_SECONDS * sample_rate))
    tail_samples = int(round(TAIL_KEEP_SECONDS * sample_rate))
    fixed_edge_samples = head_samples + tail_samples
    available_samples = int(math.floor(available * sample_rate + 1e-9))
    if available_samples <= fixed_edge_samples or len(audio) <= fixed_edge_samples:
        required = float("inf")
    else:
        # The 60/120 ms edge guards must still be present after tempo processing.
        # Apply one atempo operation to the uninterrupted middle (speech plus every
        # internal pause), then put the untouched guards back around it.
        required = (len(audio) - fixed_edge_samples) / (available_samples - fixed_edge_samples)
    details: dict[str, Any] = {
        "required": round(required, 6),
        "maximum": MAX_SPEED,
        "requested": None,
        "actual": None,
        "reason": None,
    }
    if required > MAX_SPEED + 1e-9:
        details["reason"] = "required_speed_exceeds_1.05"
        return None, details
    if len(audio) <= available_samples:
        details.update(requested=1.0, actual=1.0)
        return audio, details

    # dub.atempo intentionally treats changes below 0.5% as no-op. Request at
    # least 1.0051 when a real fit is necessary, and retry from the untouched
    # group (never stack tempo filters) if ffmpeg rounding still crosses the end.
    attempts = [max(required * 1.001, 1.0051), max(required * 1.003, 1.0051), MAX_SPEED]
    tried: set[float] = set()
    middle = audio[head_samples:len(audio) - tail_samples]
    for requested in attempts:
        requested = min(requested, MAX_SPEED)
        key = round(requested, 6)
        if key in tried:
            continue
        tried.add(key)
        rendered_middle = dub.atempo(middle, sample_rate, requested)
        rendered = np.concatenate((audio[:head_samples], rendered_middle, audio[-tail_samples:]))
        if len(rendered) <= available_samples:
            details.update(
                requested=round(requested, 6),
                actual=round(len(middle) / len(rendered_middle), 6),
            )
            return rendered, details
    details["reason"] = "atempo_output_still_exceeds_window_at_1.05"
    return None, details


def base_report(args: argparse.Namespace, data: dict[str, Any], selection_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "C3 natural whole-group assembly",
        "status": "preparing",
        "source": {
            "script": str(dub.SCRIPT),
            "script_sha256": sha256_file(Path(dub.SCRIPT)),
            "selection": str(selection_path),
            "selection_sha256": sha256_file(selection_path),
            "takes": str(Path(args.takes).resolve()),
            "group_count": len(data["groups"]),
            "cue_count": sum(len(group["lines"]) for group in data["groups"]),
            "script_duration_seconds": float(data["duration"]),
        },
        "policy": {
            "whole_group_only": True,
            "per_cue_cut_or_reorder": False,
            "head_keep_seconds": HEAD_KEEP_SECONDS,
            "tail_keep_seconds": TAIL_KEEP_SECONDS,
            "last_cue_end_grace_seconds": END_GRACE_SECONDS,
            "next_group_gap_seconds": NEXT_GAP_SECONDS,
            "max_whole_group_speed": MAX_SPEED,
            "median_speech_rms_target_db": TARGET_MEDIAN_RMS_DB,
            "group_rms_tolerance_db": GROUP_RMS_TOLERANCE_DB,
            "global_peak_limit": PEAK_LIMIT,
        },
        "claims": {
            "script_text_rewritten": False,
            "exact_55_cue_alignment_verified": False,
            "asr_verified": False,
            "human_naturalness_approved": False,
        },
        "limitations": [
            "仅按整组首条字幕起点落位；55 条 cue 的窗口被记录，但未逐条精确对齐。",
            "未运行 ASR，不能由本报告证明选条逐字正确。",
            "未进行人工试听，不能由机器报告证明自然度、情感或最终听感合格。",
        ],
        "needs_rerecord": [],
        "groups": [],
        "output": None,
    }


def write_reports(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    passed = report["status"] == "assembled_machine_checks_passed"
    lines = [
        "# C3 自然整组配音报告",
        "",
        f"- 状态：{'机器拼接检查通过' if passed else '拒绝合成，需重录超时组'}",
        f"- 组数 / cue 数：{report['source']['group_count']} / {report['source']['cue_count']}",
        "- 拼接方式：整组只裁首尾空白，保留组内停顿；不逐字幕切开或重排。",
        "- 边界：本报告未验证 55 条 cue 精确对齐，未运行 ASR，也未完成人工自然度与听感验收。",
        "",
    ]
    if report["needs_rerecord"]:
        lines += ["## 需重录", ""]
        for item in report["needs_rerecord"]:
            lines.append(
                f"- 组 {item['group']}：需 {item['required_speed']:.4f}x，超过 1.05x 上限（{item['reason']}）。"
            )
        lines.append("")
    if report.get("loudness"):
        loudness = report["loudness"]
        lines += [
            "## 音量",
            "",
            f"- 原始各组语音 RMS 中位数：{loudness['source_median_speech_rms_db']:.2f} dB",
            f"- 全组统一基础增益：{loudness['uniform_base_gain_db']:+.2f} dB",
            f"- 全轨统一峰值安全增益：{loudness['global_peak_safety_gain_db']:+.2f} dB",
            "- 仅把基础增益后的离群组修正到 -20 dB ±2 dB；句内和组内动态未归一化。",
            "",
        ]
    if report["output"]:
        output = report["output"]
        lines += [
            "## 输出",
            "",
            f"- 文件：`{output['file']}`",
            f"- 格式：{output['sample_rate']} Hz / {output['subtype']} / {output['channels']} 声道 / {output['duration_seconds']:.3f} 秒",
            f"- SHA256：`{output['sha256']}`",
            "",
        ]

    lines += [
        "## 逐组审计",
        "",
        "| 组 | 原长 | 裁后 | 基础增益 | 个别修正 | 变速 | 实际位置 | 硬窗口 | 自然停顿 |",
        "|---|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for item in report["groups"]:
        tempo = item["tempo"]["actual"]
        position = item.get("actual_position")
        position_text = "未落位" if position is None else f"{position['start']:.3f}–{position['end']:.3f}"
        lines.append(
            f"| {item['group']} | {item['trim']['source_duration_seconds']:.3f}s | "
            f"{item['trim']['trimmed_duration_seconds']:.3f}s | {item['gain']['uniform_base_db']:+.2f}dB | "
            f"{item['gain']['individual_correction_db']:+.2f}dB | "
            f"{'—' if tempo is None else f'{tempo:.4f}x'} | {position_text} | "
            f"{item['group_window']['start']:.3f}–{item['group_window']['end']:.3f} | "
            f"{item['natural_pauses_before_speed']['count']}处/{item['natural_pauses_before_speed']['total_seconds']:.2f}s |"
        )
    lines += [
        "",
        "每个 cue 窗口、首尾裁切量、停顿区间、选条种子、生成参数、参考与源音频 SHA 均保存在 `report.json`。",
        "",
    ]
    (output_dir / "配音报告.md").write_text("\n".join(lines), encoding="utf-8")


def assemble(args: argparse.Namespace) -> int:
    data = dub.load_script(dub.SCRIPT)
    groups = data["groups"]
    group_ids = [group["id"] for group in groups]
    selection_path = Path(args.selection).resolve()
    takes_dir = Path(args.takes).resolve()
    output_dir = Path(args.output_dir).resolve()
    selection = load_selection(selection_path, takes_dir, group_ids)
    report = base_report(args, data, selection_path)

    prepared: list[dict[str, Any]] = []
    source_rms_values = []
    for index, group in enumerate(groups):
        group_id = group["id"]
        selected = selection[group_id]
        audio, input_sr, input_channels, input_frames = read_mono(selected["path"])
        trimmed, trim = trim_outer_silence(audio, TARGET_SR)
        window = hard_window(data, index)
        available = float(window["available_seconds"])
        # Preserve generated onset, breath, and release whenever the complete
        # take already fits its window. Only trim outer silence when needed.
        if len(audio) <= int(math.floor(available * TARGET_SR + 1e-9)):
            trimmed = audio
            mask = dub.voiced(audio, TARGET_SR)
            active = np.flatnonzero(mask)
            hop = int(TARGET_SR * dub.FRAME)
            trim = {
                "source_duration_seconds": round_seconds(len(audio) / TARGET_SR),
                "source_trim_start_seconds": 0.0, "source_trim_end_seconds": 0.0,
                "added_head_padding_seconds": 0.0, "added_tail_padding_seconds": 0.0,
                "preserved_head_seconds": round_seconds(int(active[0]) * hop / TARGET_SR),
                "preserved_tail_seconds": round_seconds((len(audio) - min(len(audio), (int(active[-1])+1) * hop)) / TARGET_SR),
                "trimmed_duration_seconds": round_seconds(len(audio) / TARGET_SR),
                "policy": "preserve_full_take_if_it_fits",
            }
        else:
            trim["policy"] = "trim_outer_silence_to_fit"
        source_rms_db = float(dub.speech_rms_db(trimmed, TARGET_SR))
        if not math.isfinite(source_rms_db) or source_rms_db <= -89.0:
            raise SystemExit(f"组 {group_id} 无法计算有效语音 RMS")
        window = hard_window(data, index)
        available = float(window["available_seconds"])
        rendered, tempo = tempo_to_fit(trimmed, TARGET_SR, available)
        item = {
            "group": group_id,
            "script_text": group["text"],
            "cue_windows": cue_windows(group),
            "group_window": window,
            "selection": {
                "file": selected["file"],
                "seed": selected["seed"],
                "reference_sha256": selected["reference_sha256"],
                "gen": selected["gen"],
                "source_wave_sha256": selected["source_sha256"],
            },
            "input_wave": {
                "sample_rate": input_sr,
                "channels": input_channels,
                "frames": input_frames,
                "decoded_to_mono_48000": input_sr != TARGET_SR or input_channels != 1,
            },
            "trim": trim,
            "natural_pauses_before_speed": natural_pauses(trimmed, TARGET_SR),
            "natural_pauses_after_speed": None if rendered is None else natural_pauses(rendered, TARGET_SR),
            "source_speech_rms_db": round(source_rms_db, 6),
            "tempo": tempo,
            "gain": {},
            "actual_position": None,
        }
        if rendered is None:
            report["needs_rerecord"].append({
                "group": group_id,
                "required_speed": tempo["required"],
                "maximum_speed": MAX_SPEED,
                "reason": tempo["reason"],
            })
        prepared.append({"item": item, "audio": rendered})
        source_rms_values.append(source_rms_db)

    source_median = float(np.median(source_rms_values))
    uniform_gain_db = TARGET_MEDIAN_RMS_DB - source_median
    for prepared_group in prepared:
        item = prepared_group["item"]
        after_uniform = item["source_speech_rms_db"] + uniform_gain_db
        bounded = min(
            TARGET_MEDIAN_RMS_DB + GROUP_RMS_TOLERANCE_DB,
            max(TARGET_MEDIAN_RMS_DB - GROUP_RMS_TOLERANCE_DB, after_uniform),
        )
        correction = bounded - after_uniform
        item["gain"] = {
            "uniform_base_db": round(uniform_gain_db, 6),
            "speech_rms_after_uniform_db": round(after_uniform, 6),
            "individual_correction_db": round(correction, 6),
            "planned_speech_rms_before_global_peak_safety_db": round(bounded, 6),
            "global_peak_safety_db": None,
            "total_applied_db": None,
            "final_speech_rms_db": None,
        }

    report["groups"] = [prepared_group["item"] for prepared_group in prepared]
    report["loudness"] = {
        "source_median_speech_rms_db": round(source_median, 6),
        "uniform_base_gain_db": round(uniform_gain_db, 6),
        "group_rms_allowed_range_after_base_db": [-22.0, -18.0],
        "global_peak_before_safety": None,
        "global_peak_safety_gain_db": 0.0,
        "global_peak_after_safety": None,
    }

    if report["needs_rerecord"]:
        report["status"] = "rejected_needs_rerecord"
        write_reports(output_dir, report)
        print("拒绝合成：以下组需要超过 1.05x 的整组变速，请重录：")
        for item in report["needs_rerecord"]:
            print(f"  {item['group']}: {item['required_speed']:.4f}x")
        print(f"报告：{output_dir / 'report.json'}")
        return 2

    track = np.zeros(int(round(float(data["duration"]) * TARGET_SR)), dtype=np.float64)
    intervals: list[tuple[int, int, str]] = []
    for prepared_group in prepared:
        item = prepared_group["item"]
        audio = prepared_group["audio"]
        assert audio is not None
        gain_db = item["gain"]["uniform_base_db"] + item["gain"]["individual_correction_db"]
        audio = audio * db_to_gain(gain_db)
        start_sample = int(round(item["group_window"]["start"] * TARGET_SR))
        end_sample = start_sample + len(audio)
        hard_end_sample = int(math.floor(item["group_window"]["end"] * TARGET_SR + 1e-9))
        if end_sample > hard_end_sample or end_sample > len(track):
            raise RuntimeError(f"组 {item['group']} 在最终落位时越过硬窗口，未写出音轨")
        if intervals and start_sample < intervals[-1][1]:
            raise RuntimeError(f"组 {item['group']} 与组 {intervals[-1][2]} 重叠，未写出音轨")
        track[start_sample:end_sample] = audio
        intervals.append((start_sample, end_sample, item["group"]))
        item["actual_position"] = {
            "start": round_seconds(start_sample / TARGET_SR),
            "end": round_seconds(end_sample / TARGET_SR),
            "duration_seconds": round_seconds(len(audio) / TARGET_SR),
        }

    peak_before = float(np.max(np.abs(track)))
    safety_gain = min(1.0, PEAK_LIMIT / peak_before) if peak_before > 0 else 1.0
    safety_db = 20.0 * math.log10(safety_gain)
    track *= safety_gain
    peak_after = float(np.max(np.abs(track)))

    for prepared_group in prepared:
        item = prepared_group["item"]
        total_gain_db = (
            item["gain"]["uniform_base_db"]
            + item["gain"]["individual_correction_db"]
            + safety_db
        )
        item["gain"]["global_peak_safety_db"] = round(safety_db, 6)
        item["gain"]["total_applied_db"] = round(total_gain_db, 6)
        item["gain"]["final_speech_rms_db"] = round(
            item["gain"]["planned_speech_rms_before_global_peak_safety_db"] + safety_db, 6
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / WAVE_NAME
    temporary_path = output_dir / (WAVE_NAME + ".tmp")
    with temporary_path.open("wb") as handle:
        sf.write(handle, track, TARGET_SR, format="WAV", subtype="PCM_24")
    written = sf.info(str(temporary_path))
    expected_frames = int(round(float(data["duration"]) * TARGET_SR))
    if (
        written.samplerate != TARGET_SR
        or written.channels != 1
        or written.subtype != "PCM_24"
        or written.frames != expected_frames
    ):
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"输出格式验证失败：{written}")
    temporary_path.replace(output_path)

    report["loudness"].update({
        "global_peak_before_safety": round(peak_before, 9),
        "global_peak_safety_gain_db": round(safety_db, 6),
        "global_peak_after_safety": round(peak_after, 9),
    })
    report["checks"] = {
        "all_24_groups_present": len(intervals) == len(groups) == 24,
        "all_groups_inside_hard_window": True,
        "no_group_overlap": True,
        "no_placed_audio_truncated": True,
        "duration_exactly_178_seconds": written.frames == 178 * TARGET_SR,
        "format_48000_mono_pcm24": written.samplerate == TARGET_SR and written.channels == 1 and written.subtype == "PCM_24",
        "peak_at_or_below_0_89": peak_after <= PEAK_LIMIT + 1e-12,
    }
    if not all(report["checks"].values()):
        raise RuntimeError(f"输出机器检查失败：{report['checks']}")
    report["status"] = "assembled_machine_checks_passed"
    report["output"] = {
        "file": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "duration_seconds": written.frames / written.samplerate,
        "frames": written.frames,
        "sample_rate": written.samplerate,
        "channels": written.channels,
        "subtype": written.subtype,
    }
    write_reports(output_dir, report)
    print(f"人声轨：{output_path}")
    print(f"报告：{output_dir / 'report.json'}")
    print("机器拼接检查通过；仍需人工试听，且未验证 55 条 cue 精确对齐。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", default=str(DEFAULT_SELECTION), help="选条 JSON")
    parser.add_argument("--takes", default=str(DEFAULT_TAKES), help="选中 WAV 所在目录")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="自然拼接输出目录")
    return assemble(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
