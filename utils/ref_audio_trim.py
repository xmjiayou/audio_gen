# -*- coding: utf-8 -*-
"""
ref_audio_trim.py
- 若参考音频长度 > max_sec（默认 60s），仅保留前 max_sec 秒，并写入缓存文件。
- 若长度 <= max_sec，则直接返回原始路径（不复制、不改动）。
- 仅依赖: soundfile, numpy（通常你的环境已有）

用法（命令行）:
    python ref_audio_trim.py --in path/to/reference.wav --out path/to/trimmed.wav
    # 如果不指定 --out，将在 cache_dir 里生成一个带哈希的裁剪文件并打印路径

在主脚本中作为函数用:
    from ref_audio_trim import preprocess_reference
    ref_path = preprocess_reference("aigc_speech_generation_tasks/reference_1.wav",
                                    max_sec=60, cache_dir="cache/ref_trim")
"""

import argparse
import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


def _hash_sig(p: Path) -> str:
    """用文件路径 + size + mtime 生成稳定签名，避免同名文件冲突。"""
    st = p.stat()
    raw = f"{str(p.resolve())}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def _trim_to_wav(in_path: Path, out_path: Path, max_sec: float = 60.0) -> None:
    """仅写前 max_sec 秒到 out_path（覆盖写）。保留原通道数/采样率。"""
    with sf.SoundFile(in_path, mode="r") as f_in:
        sr = f_in.samplerate
        channels = f_in.channels
        subtype = f_in.subtype or "PCM_16"
        format_ = "WAV"  # 输出统一写 WAV，最稳

        max_frames = int(round(max_sec * sr))

        with sf.SoundFile(out_path, mode="w", samplerate=sr, channels=channels,
                          subtype=subtype, format=format_) as f_out:
            frames_left = max_frames
            block = 8192
            while frames_left > 0:
                to_read = min(block, frames_left)
                data = f_in.read(to_read, dtype="float32", always_2d=True)
                if data.size == 0:
                    break  # 源文件读完
                f_out.write(data)
                frames_left -= data.shape[0]


def cut_reference(in_path: str,
                         max_sec: float = 60.0,
                         cache_dir: str = "cache/ref_trim",
                         force_wav_ext: bool = True) -> str:
    """
    入口函数：返回用于后续 TTS 的参考音频路径（可能是原文件，也可能是缓存裁剪文件）。

    参数
    - in_path: 原始参考音频路径
    - max_sec: 长度阈值（秒）
    - cache_dir: 裁剪结果的缓存目录
    - force_wav_ext: 缓存文件一律写成 .wav（更通用、最稳）

    返回
    - str: 可直接用于后续流程的音频文件路径
    """
    src = Path(in_path)
    if not src.exists():
        print(f"[ref_audio_trim] WARN: input not found: {src}")
        return str(src)

    try:
        # 快速读取总时长
        with sf.SoundFile(src, mode="r") as f:
            sr = f.samplerate
            frames = len(f)
            dur = frames / float(sr)

        if dur <= max_sec + 1e-6:
            # 不裁剪，直接返回原路径
            return str(src)

        # 需要裁剪，按签名缓存，避免重复计算
        cache_root = Path(cache_dir)
        _ensure_dir(cache_root)

        sig = _hash_sig(src)
        base = src.stem
        out_name = f"{base}__trim{int(max_sec)}s__{sig}.wav" if force_wav_ext else f"{base}__trim{int(max_sec)}s__{sig}{src.suffix}"
        out_path = cache_root / out_name

        if not out_path.exists():
            _trim_to_wav(src, out_path, max_sec=max_sec)
            print(f"[ref_audio_trim] trimmed > {max_sec:.0f}s: {src.name} ({dur:.2f}s) -> {out_path.name}")

        return str(out_path)

    except Exception as e:
        print(f"[ref_audio_trim] ERROR while trimming {src}: {e}")
        # 失败时兜底返回原文件
        return str(src)


def main():
    ap = argparse.ArgumentParser(description="Trim reference audio to first N seconds if too long.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input reference audio (e.g., reference.wav)")
    ap.add_argument("--out", dest="out_path", default=None,
                    help="Optional output path; if omitted, write to cache dir and print the path.")
    ap.add_argument("--max_sec", type=float, default=60.0, help="Max seconds to keep (default: 60)")
    ap.add_argument("--cache_dir", type=str, default=".cache/ref_trim",
                    help="Cache directory for trimmed files (default: .cache/ref_trim)")
    args = ap.parse_args()

    if args.out_path:
        # 写到指定 out_path（覆盖）。若无需裁剪则拷贝前 max_sec 长度（或直接复制整段? 这里统一只写前 N 秒）
        src = Path(args.in_path)
        if not src.exists():
            raise SystemExit(f"Input not found: {src}")

        with sf.SoundFile(src, mode="r") as f:
            dur = len(f) / float(f.samplerate)
        dst = Path(args.out_path)
        _ensure_dir(dst.parent)
        # 若原始 <= max_sec，也只写原长度（不多写）
        _trim_to_wav(src, dst, max_sec=args.max_sec)
        print(str(dst))
    else:
        out = preprocess_reference(args.in_path, max_sec=args.max_sec, cache_dir=args.cache_dir)
        print(out)


if __name__ == "__main__":
    main()
