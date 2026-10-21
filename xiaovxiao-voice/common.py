"""哓v哓 声音工具包的公共代码：模型封装、描述拼接、清单与评分读写、批量生成。"""
import argparse
import hashlib
import inspect
import json
import random
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

import specs

# Windows 控制台/重定向时避免中文和符号打印报错
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = Path(__file__).resolve().parent
MODEL_TAGS = {"voxcpm2": "vox", "qwen3": "qwen", "moss": "moss"}
MANIFEST = "manifest.jsonl"
RATINGS = "ratings.json"


# ---------- 描述 ----------

def build_desc(slots, lang):
    parts = []
    for key in specs.SLOT_ORDER:
        _, zh, en = specs.SLOTS[key][slots[key]]
        phrase = zh if lang == "zh" else en
        if phrase:
            parts.append(phrase)
    suffix = specs.DESC_SUFFIX[0 if lang == "zh" else 1]
    if suffix:
        parts.append(suffix)
    if lang == "zh":
        return "，".join(parts) + "。"
    text = ", ".join(parts) + "."
    return text[0].upper() + text[1:]


def slots_zh(slots):
    return {specs.SLOT_LABELS[k]: specs.SLOTS[k][slots[k]][0] for k in specs.SLOT_ORDER if k in slots}


def slots_valid(slots):
    """旧版设定生成的条目，槽位可能和现在的 specs.py 对不上。"""
    return bool(slots) and all(k in slots and slots[k] in specs.SLOTS[k] for k in specs.SLOT_ORDER)


def clip_id(round_name, model, *parts):
    text = "|".join(json.dumps(p, sort_keys=True) if isinstance(p, dict) else str(p) for p in (model,) + parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:6]
    return f"{round_name}_{MODEL_TAGS[model]}_{digest}"


# ---------- 文件读写 ----------

def read_wav(path):
    with open(path, "rb") as f:
        audio, sr = sf.read(f, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def write_wav(path, audio, sr):
    audio = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    with open(path, "wb") as f:
        sf.write(f, audio, sr, format="WAV", subtype="PCM_16")


def read_manifest(folder):
    path = Path(folder) / MANIFEST
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def append_manifest(folder, entry):
    with open(Path(folder) / MANIFEST, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def copy_clip(entry, out_dir, suffix, **meta):
    """把上一轮的某条复制进新一轮，方便在试音台里和新生成的对照着听。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    new_id = f"{entry['id']}__{suffix}"
    if new_id in {e["id"] for e in read_manifest(out_dir)}:
        return
    shutil.copy2(Path(entry["dir"]) / entry["file"], out_dir / f"{new_id}.wav")
    keep = ("model", "archetype", "slots", "slots_zh", "desc_lang", "desc", "text", "seed", "seed_used",
            "gen", "gen_label", "sr", "duration")
    record = {k: entry[k] for k in keep if k in entry}
    record.update(id=new_id, file=f"{new_id}.wav", round=out_dir.name, parent=entry["id"], **meta)
    append_manifest(out_dir, record)


def read_ratings(folder):
    path = Path(folder) / RATINGS
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("ratings", data)


def load_rounds(folders):
    """读取若干轮的清单并合并评分。每条带上 dir、score、tags、note。"""
    entries = []
    for folder in folders:
        folder = Path(folder)
        if not folder.is_dir():
            sys.exit(f"找不到文件夹：{folder}")
        ratings = read_ratings(folder)
        if not ratings:
            print(f"提示：{folder} 里没有 {RATINGS}（还没评分，或者评分文件还在「下载」文件夹里）")
        for e in read_manifest(folder):
            r = ratings.get(e["file"], {})
            entries.append({**e, "dir": str(folder), "score": r.get("score") or 0,
                            "tags": r.get("tags") or [], "note": r.get("note") or ""})
    return entries


def top_rated(entries, n, min_score, key=lambda e: e["desc"]):
    """按分数从高到低取前 n 条，key 相同的只留最高分那条。"""
    good_tags = lambda e: sum(t.startswith("✓") for t in e["tags"])
    ranked = sorted((e for e in entries if e["score"] >= min_score),
                    key=lambda e: (-e["score"], -good_tags(e)))
    picks, seen = [], set()
    for e in ranked:
        k = key(e)
        if k not in seen:
            seen.add(k)
            picks.append(e)
        if len(picks) >= n:
            break
    return picks


# ---------- 模型 ----------

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _strip_parens(text):
    return re.sub(r"[()（）]", "", text or "").strip()


class VoxCPM2Backend:
    """VoxCPM2：造声用「(描述)台词」，克隆支持「(语气)台词」。"""

    def __init__(self, model_path=None, compile=False):
        from voxcpm import VoxCPM
        self.model = VoxCPM.from_pretrained(model_path or "openbmb/VoxCPM2",
                                            load_denoiser=False, optimize=compile)
        self.sr = self.model.tts_model.sample_rate
        # voxcpm 2.0.3 的 generate 不接受 seed，较新的版本接受；两种都兼容
        try:
            self.takes_seed = "seed" in inspect.signature(self.model._generate).parameters
        except (AttributeError, TypeError, ValueError):
            self.takes_seed = False

    def _generate(self, seed, gen, **kwargs):
        seed_everything(seed)
        if self.takes_seed:
            kwargs["seed"] = seed
        wav = self.model.generate(cfg_value=gen.get("cfg", 2.0), inference_timesteps=gen.get("steps", 10), **kwargs)
        # 若模型暴露重试后实际使用的种子则记录，否则保留本次初始种子。
        used = getattr(self.model.tts_model, "last_successful_seed", None)
        return wav, self.sr, seed if used is None else int(used)

    def design(self, desc, text, seed, gen):
        return self._generate(seed, gen, text=f"({_strip_parens(desc)}){text}")

    def clone(self, ref_wav, ref_text, text, seed, gen, style=""):
        if style:
            return self._generate(seed, gen, text=f"({_strip_parens(style)}){text}", reference_wav_path=str(ref_wav))
        return self._generate(seed, gen, text=text, prompt_wav_path=str(ref_wav),
                              prompt_text=ref_text, reference_wav_path=str(ref_wav))


class Qwen3Backend:
    """Qwen3-TTS：造声用 VoiceDesign 模型，克隆用 Base 模型（不支持语气控制）。"""

    DEFAULTS = {"design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"}

    def __init__(self, kind, model_path=None, flash_attn=False):
        import torch
        from qwen_tts import Qwen3TTSModel
        self.model = Qwen3TTSModel.from_pretrained(
            model_path or self.DEFAULTS[kind],
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if flash_attn else None,
        )
        self._prompts = {}

    def design(self, desc, text, seed, gen):
        seed_everything(seed)
        wavs, sr = self.model.generate_voice_design(text=text, language="Chinese", instruct=desc, **gen)
        return wavs[0], sr, seed

    def clone(self, ref_wav, ref_text, text, seed, gen, style=""):
        key = str(ref_wav)
        if key not in self._prompts:
            self._prompts[key] = self.model.create_voice_clone_prompt(ref_audio=key, ref_text=ref_text)
        seed_everything(seed)
        wavs, sr = self.model.generate_voice_clone(text=text, language="Chinese",
                                                   voice_clone_prompt=self._prompts[key], **gen)
        return wavs[0], sr, seed


class MossBackend:
    """MOSS-VoiceGenerator：只能造声（描述 + 台词），主打真人感。克隆请用另外两个模型。"""

    def __init__(self, model_path=None, codec_path=None):
        import torch
        from transformers import AutoModel, AutoProcessor
        # 官方示例：关掉有问题的 cuDNN SDPA，其它注意力实现保留
        torch.backends.cuda.enable_cudnn_sdp(False)
        path = model_path or "OpenMOSS-Team/MOSS-VoiceGenerator"
        extra = {"codec_path": codec_path} if codec_path else {}
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, normalize_inputs=True, **extra)
        self.processor.audio_tokenizer = self.processor.audio_tokenizer.to("cuda")
        self.model = AutoModel.from_pretrained(path, trust_remote_code=True, attn_implementation="sdpa",
                                               torch_dtype=torch.bfloat16).to("cuda").eval()
        self.sr = int(getattr(self.processor.model_config, "sampling_rate", 24000))

    def design(self, desc, text, seed, gen):
        import torch
        seed_everything(seed)
        batch = self.processor([[self.processor.build_user_message(text=text, instruction=desc)]], mode="generation")
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=batch["input_ids"].to("cuda"),
                attention_mask=batch["attention_mask"].to("cuda"),
                max_new_tokens=4096,
                # 官方推荐的解码参数（模型对这些很敏感）
                audio_temperature=gen.get("temperature", 1.5),
                audio_top_p=0.6,
                audio_top_k=50,
                audio_repetition_penalty=1.1,
            )
        messages = self.processor.decode(outputs)
        if not messages or messages[0] is None:
            raise RuntimeError("模型没有返回可解码的音频")
        audio = messages[0].audio_codes_list[0]
        if hasattr(audio, "detach"):
            audio = audio.detach().float().cpu().numpy()
        return np.asarray(audio, dtype=np.float32).reshape(-1), self.sr, seed

    def clone(self, *args, **kwargs):
        raise SystemExit("MOSS-VoiceGenerator 只能造声，克隆测试请用 --model voxcpm2 或 qwen3")


def add_model_args(parser, default_out):
    parser.add_argument("--model", choices=sorted(MODEL_TAGS), required=True,
                        help="用哪个模型生成（需要在对应的 Python 环境里运行）")
    parser.add_argument("--model-path", help="模型本地路径（已下载到本地时用；默认自动从 Hugging Face 下载）")
    parser.add_argument("--out", default=default_out, help=f"输出文件夹（默认 {default_out}）")
    parser.add_argument("--dry-run", action="store_true", help="只列出要生成什么，不加载模型")
    parser.add_argument("--compile", action="store_true", help="VoxCPM2：尝试 torch.compile 加速（Windows 上可能不可用）")
    parser.add_argument("--flash-attn", action="store_true", help="Qwen3：使用 flash-attn（需已安装）")
    parser.add_argument("--cfg", help="VoxCPM2：引导强度，官方建议 1.0~3.0（默认 2.0）。越低越自由、越高越贴描述。"
                                      "逗号分隔可一次对比多个，如 1.5,2.0")
    parser.add_argument("--steps", help="VoxCPM2：推理步数，官方建议 4~30（默认 10）。越大细节越好、越慢。可写多个，如 10,20")
    parser.add_argument("--temperature", help="Qwen3 / MOSS：采样温度（默认用模型自带设置，MOSS 官方推荐 1.5）。可写多个，如 0.9,1.0")
    parser.add_argument("--codec-path", help="MOSS：MOSS-Audio-Tokenizer 的本地路径（从 ModelScope 下到本地时用）")


def _numbers(text):
    return [float(x) for x in str(text).split(",") if x.strip()]


def gen_grid(args, inherit=None):
    """生成参数的所有组合。命令行写了就用命令行的（逗号分隔即网格），没写就沿用 inherit（上一轮那条的参数）。"""
    inherit = inherit or {}
    if args.model == "voxcpm2":
        cfgs = _numbers(args.cfg) if args.cfg else [inherit.get("cfg", 2.0)]
        steps = [int(x) for x in _numbers(args.steps)] if args.steps else [inherit.get("steps", 10)]
        return [{"cfg": c, "steps": s} for c in cfgs for s in steps]
    if args.temperature:
        return [{"temperature": t} for t in _numbers(args.temperature)]
    return [{"temperature": inherit["temperature"]}] if "temperature" in inherit else [{}]


def gen_label(gen):
    parts = []
    if "cfg" in gen:
        parts.append(f"cfg{gen['cfg']:g}")
    if "steps" in gen:
        parts.append(f"步数{gen['steps']}")
    if "temperature" in gen:
        parts.append(f"温度{gen['temperature']:g}")
    return " ".join(parts) or "默认参数"


def make_backend(args, kind):
    print(f"正在加载 {args.model}（第一次会下载模型，请耐心等待）…")
    if args.model == "voxcpm2":
        return VoxCPM2Backend(args.model_path, compile=args.compile)
    if args.model == "moss":
        return MossBackend(args.model_path, args.codec_path)
    return Qwen3Backend(kind, args.model_path, flash_attn=args.flash_attn)


# ---------- 批量生成 ----------

def _label(meta):
    label = " / ".join(str(meta[k]) for k in ("archetype", "variant", "line") if meta.get(k))
    if meta.get("desc_lang") in ("zh", "en"):
        label += f" [{meta['desc_lang']}]"
    if meta.get("gen"):
        label += f" {meta['gen_label']}"
    return label


def run_jobs(jobs, out_dir, load_backend, dry_run=False):
    """jobs: [{"meta": {...含 id}, "run": backend -> (wav, sr, seed_used)}]。已生成的自动跳过，可随时中断续跑。"""
    out_dir = Path(out_dir)
    unique = {}
    for job in jobs:
        unique.setdefault(job["meta"]["id"], job)
    jobs = list(unique.values())

    done = {e["id"] for e in read_manifest(out_dir)}
    todo = [j for j in jobs if not (j["meta"]["id"] in done and (out_dir / f"{j['meta']['id']}.wav").exists())]
    print(f"计划 {len(jobs)} 条：已完成 {len(jobs) - len(todo)}，本次生成 {len(todo)} → {out_dir}")

    if dry_run:
        for j in todo:
            print(f"  {j['meta']['id']}  {_label(j['meta'])}  seed={j['meta'].get('seed')}")
        return
    if not todo:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    backend = load_backend()
    started, failed = time.time(), 0
    for i, job in enumerate(todo, 1):
        meta = dict(job["meta"])
        wav_path = out_dir / f"{meta['id']}.wav"
        t0 = time.time()
        try:
            wav, sr, seed_used = job["run"](backend)
        except KeyboardInterrupt:
            print("\n已中断。重新运行同一条命令会从这里继续。")
            raise SystemExit(1)
        except Exception as exc:  # 单条失败不影响整批
            failed += 1
            print(f"[{i}/{len(todo)}] ✗ {meta['id']} 失败：{exc}")
            continue
        write_wav(wav_path, wav, sr)
        meta.update(seed_used=seed_used, sr=sr, duration=round(len(np.asarray(wav).reshape(-1)) / sr, 2),
                    created=datetime.now().isoformat(timespec="seconds"))
        append_manifest(out_dir, meta)
        elapsed = time.time() - started
        eta = elapsed / i * (len(todo) - i)
        print(f"[{i}/{len(todo)}] {meta['id']}  {_label(meta)}  {meta['duration']}s 音频，用时 {time.time() - t0:.1f}s，"
              f"预计还要 {eta / 60:.1f} 分钟")
    print(f"完成：成功 {len(todo) - failed} 条，失败 {failed} 条。")
    if failed < len(todo):
        import measure
        if measure.available():
            measure.measure_folder(out_dir)
        else:
            print("（没装 praat-parselmouth，跳过声学测量。装上后可运行 python measure.py 补测）")
        print(f"用试音台（rater.html）打开 {out_dir} 开始听。")
    if failed:
        print("失败的条目重新运行同一条命令会再试一次。")


def new_parser(description):
    return argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
