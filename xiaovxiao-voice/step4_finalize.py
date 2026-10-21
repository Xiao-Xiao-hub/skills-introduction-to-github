"""第四步 · 定妆：把选中的那一条导出成「哓v哓_定妆音」，以后所有配音都用它当参考音频。

做的事：去掉首尾静音、统一音量、写好对应文字稿，再存一份声音档案（描述、模型、种子），方便以后追溯和重现。

例子：
  python step4_finalize.py rounds/r2/r2_vox_1a2b3c.wav
  python step4_finalize.py rounds/r2/r2_vox_1a2b3c.wav --name 哓v哓 --out final
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from common import read_manifest, read_wav, write_wav


def trim_silence(audio, sr, threshold_db=-40.0, pad=0.15):
    frame = max(1, int(sr * 0.02))
    n = len(audio) // frame
    if n == 0:
        return audio
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    level = 20 * np.log10(rms / (rms.max() + 1e-12))
    voiced = np.where(level > threshold_db)[0]
    if len(voiced) == 0:
        return audio
    start = max(0, voiced[0] * frame - int(pad * sr))
    end = min(len(audio), (voiced[-1] + 1) * frame + int(pad * sr))
    return audio[start:end]


def normalize(audio, target_rms_db=-20.0, peak_db=-1.0):
    rms = np.sqrt(np.mean(audio ** 2) + 1e-12)
    gain = 10 ** (target_rms_db / 20) / rms
    peak = np.abs(audio).max() * gain
    limit = 10 ** (peak_db / 20)
    if peak > limit:
        gain *= limit / peak
    return audio * gain


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", help="选中的音频文件路径")
    parser.add_argument("--name", default="哓v哓", help="角色名（用于文件名）")
    parser.add_argument("--out", default="final", help="输出文件夹（默认 final）")
    parser.add_argument("--text", help="音频对应的文字（默认从清单里读取）")
    parser.add_argument("--no-trim", action="store_true", help="不裁剪首尾静音")
    args = parser.parse_args()

    clip = Path(args.clip)
    if not clip.exists():
        raise SystemExit(f"找不到文件：{clip}")
    meta = next((e for e in read_manifest(clip.parent) if e["file"] == clip.name), {})
    text = args.text or meta.get("text")
    if not text:
        raise SystemExit("清单里找不到这条音频的文字，请用 --text 写上它说的内容。")

    audio, sr = read_wav(clip)
    if not args.no_trim:
        audio = trim_silence(audio, sr)
    audio = normalize(audio)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wav_path = out / f"{args.name}_定妆音.wav"
    write_wav(wav_path, audio, sr)
    (out / f"{args.name}_定妆音.txt").write_text(text, encoding="utf-8")
    profile = {
        "name": args.name,
        "exported": datetime.now().isoformat(timespec="seconds"),
        "source_file": str(clip),
        "duration": round(len(audio) / sr, 2),
        "sample_rate": sr,
        "text": text,
        "source": meta,
    }
    (out / f"{args.name}_声音档案.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"已导出：{wav_path}（{profile['duration']} 秒）")
    print(f"文字稿：{out / (args.name + '_定妆音.txt')}")
    if meta.get("desc"):
        print(f"它的描述：{meta['desc']}（模型 {meta.get('model')}，种子 {meta.get('seed_used', meta.get('seed'))}）")
    if not 6 <= profile["duration"] <= 20:
        print("注意：参考音频最好在 6~20 秒之间，太短音色不稳，太长有的模型会截断。")
    print("\n以后怎么用：")
    print(f"  VoxCPM2：reference_wav_path='{wav_path}'，要高保真时再加 prompt_wav_path 同一文件 + prompt_text=文字稿")
    print("  IndexTTS-2.5：WebUI 的「音色参考音频」里放这个文件")
    print("  记得把 final 文件夹备份一份——这就是哓v哓声音的「身份证」。")


if __name__ == "__main__":
    main()
