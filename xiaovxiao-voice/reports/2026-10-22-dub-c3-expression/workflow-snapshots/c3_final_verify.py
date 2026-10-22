"""C3 final whole-group extraction and (optional) sequential ASR verification.

This tool deliberately verifies whole groups, not exact alignment of 55 cues.
Use ``--extract-only`` to create the audit artifacts without importing ASR.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import soundfile as sf
import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dub
OUT = ROOT / "dub" / "out" / "c3-expression-2026-10-22" / "natural-final"
REPORT = OUT / "report.json"
WAVE = OUT / "人声轨_48k24bit.wav"
TARGET_SR = 48_000
TARGET_SECONDS = 178
CHECK_GROUPS = {"07", "18", "23a", "23b", "26-27"}
PLAN = ROOT / "reports" / "2026-10-22-dub-c3-expression" / "plan.json"
SENSEVOICE = ROOT / "models" / "SenseVoiceSmall"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm(text: str) -> str:
    """Use the explicit same-sound spelling set from c_consistency_select."""
    import re
    value = re.sub(r"[^\u4e00-\u9fffA-Za-z]", "", text).lower()
    for a, b in [("明瞭", "明了"), ("小念", "晓念"), ("小是让", "晓是让"),
                 ("要做的是", "要做的事"), ("原画", "原话"), ("原化", "原话"),
                 ("代办", "待办"), ("小黎", "小狸"), ("小离", "小狸"),
                 ("他会", "它会"), ("在反馈", "再反馈"), ("先际状态", "先记状态"),
                 ("算是另外", "算式另外"), ("sqlight", "sqlite")]:
        value = value.replace(a, b)
    return value


def fail(checks: dict[str, bool], name: str, ok: bool) -> None:
    checks[name] = bool(ok)


def extract(report: dict, wave: Path, audit: Path) -> tuple[list[dict], list[str]]:
    audit.mkdir(parents=True, exist_ok=True)
    audio, sr = sf.read(str(wave), dtype="float32", always_2d=True)
    checks: list[str] = []
    info=sf.info(wave)
    if sr != TARGET_SR or audio.shape[1] != 1 or info.subtype!='PCM_24':
        checks.append("source_format")
    if abs(len(audio)/sr-TARGET_SECONDS)>1/sr:checks.append('actual_wave_duration')
    if not np.isfinite(audio).all() or np.max(np.abs(audio))>.890001:checks.append('actual_peak_or_finite')
    if np.count_nonzero(np.abs(audio)>=.999)>0:checks.append('actual_fullscale')
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    selection_path=OUT.parent/'selection.json'
    selection=json.loads(selection_path.read_text(encoding='utf-8'))
    if report.get('source',{}).get('selection_sha256')!=sha256(selection_path):checks.append('selection_sha256')
    for name,anchor in plan['expression_bank'].items():
        if sha256(ROOT/anchor['path'])!=anchor['sha256']:checks.append('expression_anchor_changed:'+name)
    for filename,expected_sha in plan['source_hashes'].items():
        if sha256(ROOT/filename)!=expected_sha:checks.append('source_changed:'+filename)
    output = report.get("output") or {}
    if output.get("sha256") != sha256(wave):
        checks.append("source_sha256")
    if output.get("duration_seconds") is not None and abs(float(output["duration_seconds"]) - TARGET_SECONDS) > 1e-6:
        checks.append("source_duration")
    groups = report.get("groups", [])
    expected_groups=dub.load_script(dub.SCRIPT)['groups']
    if [g.get('group') for g in groups]!=[g['id'] for g in expected_groups]:checks.append('group_order')
    if len(groups) != 24:
        checks.append("group_count")
    manifest = []
    previous_end = -1.0
    for index,item in enumerate(groups):
        group = str(item.get("group", ""))
        pos = item.get("actual_position") or {}
        start, end = float(pos.get("start", -1)), float(pos.get("end", -1))
        checks_map: dict[str, bool] = {}
        fail(checks_map, "position_present", start >= 0 and end > start)
        fail(checks_map, "inside_178s", 0 <= start < end <= TARGET_SECONDS)
        fail(checks_map, "no_overlap", start >= previous_end)
        original=expected_groups[index] if index<len(expected_groups) else {}
        fail(checks_map,'original_text',item.get('script_text')==original.get('text'))
        chosen=selection.get(group,{})
        expected_bank=plan['group_bank'].get(group)
        expected_anchor=plan['expression_bank'].get(expected_bank,{})
        fail(checks_map,'current_expression_bank',chosen.get('expression_bank')==expected_bank and chosen.get('prompt_sha256')==expected_anchor.get('sha256'))
        fail(checks_map,'selection_passed',chosen.get('failed')==[])
        fail(checks_map,'selected_source_hash',item.get('selection',{}).get('source_wave_sha256')==chosen.get('sha256'))
        if original:
            limit=min(original['lines'][-1]['end']+.35,expected_groups[index+1]['lines'][0]['start']-.06 if index+1<len(expected_groups) else 178)
            fail(checks_map,'hard_window',abs(start-original['lines'][0]['start'])<=1/TARGET_SR and end<=limit+1/TARGET_SR)
        fail(checks_map,'speed',float(item.get('tempo',{}).get('requested') or 1)<=1.05)
        if not checks_map["no_overlap"]: checks.append(f"overlap:{group}")
        if not checks_map["position_present"]: checks.append(f"position:{group}")
        if not checks_map["inside_178s"]: checks.append(f"window:{group}")
        previous_end = max(previous_end, end)
        start_i, end_i = round(start * TARGET_SR), round(end * TARGET_SR)
        if 0<=start_i<end_i<=len(audio):
            fail(checks_map,'non_silent',np.max(np.abs(audio[start_i:end_i]))>.01)
        for name,passed in checks_map.items():
            if not passed:checks.append(f'{name}:{group}')
        path = audit / f"c3_g{group}.wav"
        if checks_map["position_present"] and checks_map["inside_178s"]:
            sf.write(str(path), audio[start_i:end_i, 0], TARGET_SR, subtype="PCM_24")
        lines = item.get("cue_windows", [])
        manifest.append({"id":path.stem,"profile": "c3", "group": group, "text": item.get("script_text", ""),
                         "file": path.name, "path": str(path), "start": start, "end": end,
                         "sha256": sha256(path) if path.exists() else None,
                         "cue_windows": lines, "checks": checks_map})
    (audit / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (audit / "manifest.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in manifest), encoding="utf-8")
    return manifest, checks


def run_asr(manifest: list[dict], selected: set[str]) -> list[dict]:
    """Load Paraformer, destroy it, then load SenseVoice; no expected text is passed."""
    from funasr import AutoModel
    rows = []
    paraformer = AutoModel(model="paraformer-zh", device="cuda:0", ncpu=4,
                           disable_update=True, disable_pbar=True)
    try:
        for row in manifest:
            wav, sr = sf.read(str(Path(row["path"])), dtype="float32")
            wav = resample_poly(wav, 16000, sr).astype(np.float32) if sr != 16000 else wav
            r = paraformer.generate(input=wav, fs=16000, batch_size_s=300, disable_pbar=True)
            row["paraformer"] = {k:v for k,v in r[0].items() if k in ['text','timestamp']} if r else {}
            print('Paraformer',row['group'],norm(row['paraformer'].get('text',''))==norm(row['text']),row['paraformer'].get('text',''),flush=True)
    finally:
        del paraformer
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
    sense = AutoModel(model=str(SENSEVOICE), device="cuda:0", ncpu=4,
                      disable_update=True, disable_pbar=True)
    try:
        for row in manifest:
            if row["group"] not in selected: continue
            wav, sr = sf.read(str(Path(row["path"])), dtype="float32")
            wav = resample_poly(wav, 16000, sr).astype(np.float32) if sr != 16000 else wav
            r = sense.generate(input=wav, fs=16000, language='zh',use_itn=True,disable_pbar=True)
            row["sensevoice_result"] = {k:v for k,v in r[0].items() if k in ['text','timestamp']} if r else {}
            raw = (r[0].get("text") if r else "") or ""
            import re
            raw_plain = re.sub(r"<\|[^|>]+\|>", "", raw)
            row["sensevoice_raw"] = raw
            row["sensevoice_plain"] = raw_plain
            row["transcript_normalized"] = norm(raw_plain)
            row["needs_review"] = row["transcript_normalized"] != norm(row["text"])
            print('SenseVoice',row['group'],not row['needs_review'],raw_plain,flush=True)
            rows.append(row)
    finally:
        del sense
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(REPORT)); ap.add_argument("--wave", default=str(WAVE))
    ap.add_argument("--audit-dir", default=str(OUT.parent / "audit-groups")); ap.add_argument("--extract-only", action="store_true")
    args = ap.parse_args()
    report_path, wave, audit = Path(args.report), Path(args.wave), Path(args.audit_dir)
    if not report_path.is_file() or not wave.is_file():
        print("FAIL missing report or final WAV"); return 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest, errors = extract(report, wave, audit)
    rows = []
    if not args.extract_only and not errors:
        rows = run_asr(manifest, CHECK_GROUPS)
    for row in manifest:
        expected = norm(row["text"])
        actual = norm((row.get("paraformer", {}).get("text") or "")) if rows else None
        row["paraformer_normalized"] = actual
        row["paraformer_match"] = (actual == expected) if actual is not None else None
        if actual is not None and actual != expected:
            errors.append("paraformer_text:" + row["group"])
    if not args.extract_only:
        for row in manifest:
            if row["group"] in CHECK_GROUPS and row.get("needs_review"):
                errors.append("sensevoice_text:" + row["group"])
    technical = (not args.extract_only and not errors and len(manifest) == 24 and
                 all(all(x["checks"].values()) for x in manifest))
    result = {"profile": "c3", "technical_passed": technical, "human_listening_passed": False,
              "whole_group_alignment_only": True, "exact_55_cue_alignment_verified": False,
              "model_rows": rows, "failures": errors,
              "source_sha256": sha256(wave), "source_report_sha256": sha256(report_path),
              "asr_completed":len(rows)==24,
              "source_fullscale_check":'actual_fullscale' not in errors}
    (audit / "final-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FAIL", errors if errors else "[]")
    return 0 if technical or (args.extract_only and not errors) else 1


if __name__ == "__main__":
    raise SystemExit(main())
