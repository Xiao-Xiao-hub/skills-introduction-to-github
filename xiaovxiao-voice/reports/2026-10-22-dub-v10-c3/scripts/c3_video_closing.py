"""Generate and evidence one new expressive C3 closing; source scripts remain untouched."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dub
import measure
from common import VoxCPM2Backend
from c_consistency_select import normalize_transcript

OUT = ROOT / "dub/out/c3-video-2026-10-22/closing"
PLAN = ROOT / "reports/2026-10-22-dub-c3-expression/plan.json"
SOURCE_SHA = "a0db0f1e86000b320f0d64d869c1e72122c89dacea3ba4bf3558ef530a4785b5"
TEXT = "念，是转瞬的念头。晓，是让所想渐渐明瞭。晓念。"
CUES = [(169.75, 171.85), (172.12, 174.35), (175.35, 176.25)]
WARM = "warm"
STYLES = [
    ("sincere", "真诚、温暖、带希望，像在和观众分享自己的作品；前两句逐步递进，最后晓念温柔笃定收住，不喊不煽情。"),
    ("conversational", "自然亲近、真诚温暖，有交流感和轻微情绪起伏；前两句逐步递进，最后晓念温柔笃定收住，不喊不煽情。"),
]
SEEDS = [61, 67]

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def write(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def unload(model) -> None:
    del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

def plain(text: str) -> str:
    return re.sub(r"<\|[^|>]+\|>", "", text or "").strip()

def strict_match(actual: str) -> bool:
    # The shared normalizer intentionally allows documented same-sound variants;
    # pronunciation of the brand character 晓 is an explicit additional gate.
    return normalize_transcript(TEXT) == normalize_transcript(actual)

def cue_fit(timestamp: list, raw_text: str) -> dict:
    # Paraformer normally emits one timestamp per Han character.  Surface a
    # conservative, auditable fit check; the integrator will do final slicing.
    chars = re.sub(r"[^\u4e00-\u9fff]", "", raw_text or "")
    expected = [7, 9, 2]
    result = {"timestamp_count": len(timestamp or []), "recognized_han_count": len(chars), "cues": []}
    offset = 0
    for (start, end), count in zip(CUES, expected):
        item = {"window": [start, end], "window_seconds": round(end-start, 3), "expected_characters": count}
        part = (timestamp or [])[offset:offset+count]
        offset += count
        if len(part) == count:
            speech = max(0.0, (part[-1][1] - part[0][0]) / 1000.0)
            item.update(raw_speech_seconds=round(speech, 3), needed_tempo=round(speech/(end-start), 4), passes_max_1_10=speech <= (end-start)*1.10)
        else:
            item.update(raw_speech_seconds=None, needed_tempo=None, passes_max_1_10=False)
        result["cues"].append(item)
    result["all_pass"] = all(x["passes_max_1_10"] for x in result["cues"])
    return result

def gen_batch(plan: dict, batch: int, styles: list[tuple[str, str]], seeds: list[int], synthesis_input: str = TEXT) -> list[dict]:
    profile = next(p for p in plan["profiles"] if p["id"] == WARM)
    # C3's approved warm expression is the reference.  The profiles.warm
    # calibration file is an older C2 calm anchor and is deliberately not used.
    anchor = plan["expression_bank"]["warm"]
    ref = ROOT / anchor["path"]
    if sha(ref) != anchor["sha256"]:
        raise SystemExit("approved C3 warm reference hash mismatch")
    backend = VoxCPM2Backend(str(ROOT / "models/VoxCPM2"))
    rows = []
    try:
        for style_id, control in styles:
            for seed in seeds:
                stem = f"closing_b{batch}_{style_id}_s{seed}"
                path = OUT / f"{stem}.wav"
                wav, sr, used = backend.clone(ref, anchor["text"], synthesis_input, seed, profile["gen"], control)
                sf.write(path, wav, sr, subtype="PCM_24")
                rows.append({"id": stem, "path": str(path.resolve()), "sha256": sha(path), "style_id": style_id,
                             "control": control, "seed": seed, "seed_used": used, "mode": "reference_only_plus_control",
                             "reference": str(ref.resolve()), "reference_sha256": anchor["sha256"], "reference_text":anchor["text"], "gen": profile["gen"],
                             "original_text":TEXT, "synthesis_input":synthesis_input,
                             "duration": len(wav)/sr, "sample_rate": sr, "batch": batch})
                print("generated", stem, flush=True)
    finally:
        unload(backend); del backend
    return rows

def run_asr(rows: list[dict]) -> None:
    from funasr import AutoModel
    para = AutoModel(model="paraformer-zh", device="cuda:0", ncpu=4, disable_update=True, disable_pbar=True)
    try:
        for row in rows:
            a, sr = sf.read(row["path"], dtype="float32")
            a16 = resample_poly(a, 16000, sr).astype(np.float32) if sr != 16000 else a
            r = para.generate(input=a16, fs=16000, batch_size=1, disable_pbar=True)[0]
            row["paraformer"] = {k:r.get(k) for k in ("text", "timestamp")}
            row["paraformer"]["strict_match"] = strict_match(row["paraformer"].get("text", ""))
            row["cue_fit"] = cue_fit(row["paraformer"].get("timestamp", []), row["paraformer"].get("text", ""))
            print("paraformer", row["id"], row["paraformer"]["text"], flush=True)
    finally:
        unload(para); del para
    # Only the best two primary-ASR candidates receive independent ASR.
    ranked = sorted(rows, key=lambda x: (not x["paraformer"]["strict_match"], not x["cue_fit"]["all_pass"], x["duration"]))[:2]
    sense = AutoModel(model=str(ROOT/"models/SenseVoiceSmall"), device="cuda:0", ncpu=4, disable_update=True, disable_pbar=True, trust_remote_code=False)
    try:
        for row in ranked:
            a, sr = sf.read(row["path"], dtype="float32")
            a16 = resample_poly(a, 16000, sr).astype(np.float32) if sr != 16000 else a
            r = sense.generate(input=a16, fs=16000, language="zh", use_itn=True, disable_pbar=True)[0]
            raw = r.get("text", "")
            row["sensevoice"] = {"raw": raw, "plain": plain(raw), "timestamp": r.get("timestamp")}
            row["sensevoice"]["strict_match"] = strict_match(row["sensevoice"]["plain"])
            print("sensevoice", row["id"], row["sensevoice"]["plain"], flush=True)
    finally:
        unload(sense); del sense

def cam_and_acoustic(rows: list[dict], plan: dict) -> None:
    from funasr import AutoModel
    refs = {k: ROOT/v["path"] for k,v in plan["expression_bank"].items() if k in ("warm", "soft")}
    # Existing C3 group embeddings are a stable cross-group comparison set.
    old = json.loads((ROOT/"dub/out/c3-expression-2026-10-22/audit-groups-speaker.json").read_text(encoding="utf-8"))
    peers = {x["group"]: np.asarray(x["embedding"]) for x in old if x.get("group") in ("05-06", "07", "20", "24")}
    old_close = next(x for x in old if x.get("group") == "26-27")
    model = AutoModel(model=str(ROOT/"models/CAMPlusPlus"), device="cpu", ncpu=4, disable_update=True, disable_pbar=True)
    try:
        vectors = {}
        for key, path in refs.items():
            a,sr=sf.read(path,dtype="float32"); a16=resample_poly(a,16000,sr).astype(np.float32) if sr!=16000 else a
            v=model.generate(input=a16,fs=16000,batch_size=1,disable_pbar=True)[0]["spk_embedding"].detach().cpu().numpy().reshape(-1); vectors[key]=v/np.linalg.norm(v)
        for row in rows:
            a,sr=sf.read(row["path"],dtype="float32"); a16=resample_poly(a,16000,sr).astype(np.float32) if sr!=16000 else a
            v=model.generate(input=a16,fs=16000,batch_size=1,disable_pbar=True)[0]["spk_embedding"].detach().cpu().numpy().reshape(-1); v=v/np.linalg.norm(v)
            row["camplusplus"]={"warm_cosine":float(v@vectors["warm"]),"soft_cosine":float(v@vectors["soft"]),
                                  "c3_other_groups":{k:float(v@pv) for k,pv in peers.items()}, "old_c2_closing_cosine":float(v@np.asarray(old_close["embedding"]))}
    finally:
        del model; gc.collect()
    # Pitch spread is descriptive only, never a naturalness score.
    def f0(path):
        a,sr=sf.read(path,dtype="float32"); p=measure.parselmouth.Sound(a,sampling_frequency=sr).to_pitch_ac(time_step=.01,pitch_floor=100,pitch_ceiling=500)
        z=p.selected_array["frequency"]; z=z[z>0]
        return {"f0_p10":float(np.percentile(z,10)),"f0_p90":float(np.percentile(z,90)),"f0_spread_hz":float(np.percentile(z,90)-np.percentile(z,10))} if len(z) else {}
    old_path=ROOT/"dub/out/c3-expression-2026-10-22/audit-groups/c3_g26-27.wav"
    old_metric=f0(old_path)
    for row in rows: row["acoustic"]={"candidate":f0(Path(row["path"])),"old_closing":old_metric,"note":"descriptive only; does not establish naturalness"}

def choose(rows: list[dict]) -> dict | None:
    viable=[r for r in rows if r["paraformer"]["strict_match"] and r["cue_fit"]["all_pass"] and r.get("sensevoice",{}).get("strict_match")
            and r["camplusplus"]["warm_cosine"] >= .85 and np.median(list(r["camplusplus"]["c3_other_groups"].values())) >= .85]
    if not viable: return None
    # Treat speaker cosine as a drift guard; text and timing are hard gates.
    return max(viable, key=lambda r: (r["camplusplus"]["warm_cosine"], np.median(list(r["camplusplus"]["c3_other_groups"].values()))))

def save_selected(chosen: dict | None, candidates_path: Path) -> int:
    if chosen is None:
        write(OUT/"selected.json", {"passed":False,"source_delivery_sha256":SOURCE_SHA,"reason":"No newly generated candidate passed strict raw Paraformer and SenseVoice pronunciation gates, cue-fit gate, and CAM++ identity drift gates.","candidates_json":str(candidates_path.resolve())})
        return 1
    selected={"passed":True,"source_delivery_sha256":SOURCE_SHA,"path":chosen["path"],"sha256":chosen["sha256"],"duration":chosen["duration"],"sample_rate":chosen["sample_rate"],"original_text":TEXT,"synthesis_input":chosen.get("synthesis_input",TEXT),"generation":{k:chosen[k] for k in ("mode","style_id","control","seed","seed_used","reference","reference_sha256","gen","batch")},"paraformer":chosen["paraformer"],"sensevoice":chosen["sensevoice"],"cue_fit":chosen["cue_fit"],"camplusplus":chosen["camplusplus"],"acoustic":chosen["acoustic"],"reason":"New C3 warm-reference candidate passed strict original-text pronunciation, timestamp cue-fit, and CAM++ identity-drift gates; CAM++ does not prove naturalness.","candidates_json":str(candidates_path.resolve())}
    write(OUT/"selected.json",selected)
    return 0

def pronunciation_batch(plan: dict) -> int:
    alias = "念，是转瞬的念头。小，是让所想渐渐明瞭。小念。"
    if normalize_transcript(alias) != normalize_transcript(TEXT):
        raise SystemExit("phonetic alias must normalize to original text")
    control = "自然亲近、真诚温暖，有交流感和轻微情绪起伏；前两句逐步递进，最后晓念温柔笃定收住，不喊不煽情。 句内连贯不拖字，不过度拉长停顿。"
    rows = gen_batch(plan, 3, [("pronunciation", control)], [79, 83], alias)
    for row in rows:
        row["alias_reason"] = "Only synthesis input replaces 晓 with same-pronunciation 小 to guide the rare character; original script and expected ASR remain 晓. 想 is never normalized to 晓."
    run_asr(rows); cam_and_acoustic(rows,plan)
    cpath=OUT/"candidates.json"
    old=json.loads(cpath.read_text(encoding="utf-8")) if cpath.exists() else {"source_delivery_sha256":SOURCE_SHA,"text":TEXT,"cues":CUES,"candidates":[]}
    old["candidates"].extend(rows); old["pronunciation_alias_batch"]={"alias":alias,"original":TEXT,"normalization_equal":True,"expected_asr":"original normalized text","count":2}
    write(cpath,old)
    return save_selected(choose(rows),cpath)

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    plan=json.loads(PLAN.read_text(encoding="utf-8"))
    if "--pronunciation-batch" in sys.argv:
        return pronunciation_batch(plan)
    interrupted=sorted(str(p.resolve()) for p in OUT.glob("closing_b1_*.wav"))
    if interrupted:
        write(OUT/"interrupted-old-c2-reference-batch.json", {"status":"not eligible", "paths":interrupted,
            "reason":"Stopped before analysis: the initial helper mistakenly used profiles.warm C2 calm reference, not approved C3 expression_bank.warm. Files retained as non-candidate evidence."})
    styles=[(name, text+" 句内连贯不拖字，不过度拉长停顿。") for name,text in STYLES]
    rows=gen_batch(plan,1,styles,SEEDS); run_asr(rows); cam_and_acoustic(rows,plan)
    chosen=choose(rows)
    if chosen is None:
        # One targeted second batch only: calmer word boundaries to help 晓念.
        extra=gen_batch(plan,2,[("clear_warm", "真诚温暖、简洁连贯地分享自己的作品，清楚说出晓念；前两句递进，最后温柔笃定收住。句内连贯不拖字，不过度拉长停顿。"), ("soft_hopeful", "温柔真诚、有希望，和观众面对面交流；晓字和晓念清楚自然，最后轻轻收住。句内连贯不拖字，不过度拉长停顿。")],[71,73])
        run_asr(extra); cam_and_acoustic(extra,plan); rows.extend(extra); chosen=choose(rows)
    write(OUT/"candidates.json", {"source_delivery_sha256":SOURCE_SHA,"text":TEXT,"cues":CUES,"candidates":rows})
    return save_selected(chosen,OUT/"candidates.json")

if __name__ == "__main__":
    raise SystemExit(main())
