#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RUNNER（评分模块外置版）：
- 参考音频：ref_preproc_dual 双路（embed_ref/asr_ref）
- 候选生成：IndexTTS / XTTS(缓存) / F5-CLI / Piper兜底
- 评分选优：scoring.py（WER+SIM），并把每个候选的得分写到 CSV 日志
"""
from __future__ import annotations
import os, sys, gc, zipfile, shutil, warnings, random, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm

from utils.num_cn_normalizer import normalize_text
from utils.ref_audio_trim import cut_reference
from utils.resume_utils import list_official_wavs
from utils.ref_preproc_dual import preprocess_reference_dual, DualRefCfg

from utils.synth_backends import (
    IndexTTSBackend, XTTSBackend, F5TTSCLIBackend, PiperBackend,
    xtts_available, piper_available
)

# 新的评分器
from utils.scoring import CandidateScorer


# ========= 小工具 =========
def apply_fade(y: np.ndarray, sr: int, fade_ms: int = 10) -> np.ndarray:
    n = max(1, int(sr * fade_ms / 1000))
    if len(y) < 2 * n: return y
    y = y.astype(np.float32, copy=True)
    y[:n] *= np.linspace(0, 1, n, endpoint=True).astype(np.float32)
    y[-n:] *= np.linspace(1, 0, n, endpoint=True).astype(np.float32)
    return y

def concat_crossfade(chunks: list[np.ndarray], sr: int, xfade_ms: int = 10) -> np.ndarray:
    if not chunks: return np.zeros(0, dtype=np.float32)
    xfade = int(sr * xfade_ms / 1000)
    out = chunks[0].astype(np.float32, copy=True)
    for i in range(1, len(chunks)):
        a, b = out, chunks[i].astype(np.float32, copy=False)
        if xfade > 0 and len(a) > xfade and len(b) > xfade:
            mix = a[-xfade:] * np.linspace(1, 0, xfade).astype(np.float32) + \
                  b[:xfade] * np.linspace(0, 1, xfade).astype(np.float32)
            out = np.concatenate([a[:-xfade], mix, b[xfade:]], axis=0)
        else:
            out = np.concatenate([a, b], axis=0)
    return np.clip(out, -1.0, 1.0)

def estimate_duration(text: str) -> float:
    t = str(text)
    base = 0.18 * sum(1 for c in t if c.strip())
    bonus = 0.2 * sum(1 for c in t if c in "。！？!?")
    return max(0.6, min(15.0, base + bonus))

def write_tone_placeholder(out_wav: Path, seconds=1.2, sr=16000, freq=440.0):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    y = 0.1 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    y = apply_fade(y, sr, fade_ms=50)
    sf.write(out_wav, y, sr)
    return out_wav

def ref_mimic_collage(ref_wav: Path, target_seconds: float, out_wav: Path, sr=16000):
    y, sro = sf.read(ref_wav)
    if y.ndim > 1: y = y.mean(axis=1)
    if sro != sr: y = librosa.resample(y, sro, sr)
    y = y.astype(np.float32)
    if len(y) < int(0.2 * sr):
        return write_tone_placeholder(out_wav, seconds=max(0.6, target_seconds), sr=sr)

    frm = int(0.02 * sr); hop = int(0.01 * sr)
    frames = [y[i:i + frm] for i in range(0, max(1, len(y) - frm), hop)]
    rms = np.array([np.sqrt((f ** 2).mean() + 1e-9) for f in frames])
    th = max(1e-4, np.median(rms) * 0.6)
    voiced_idx = [i for i, r in enumerate(rms) if r >= th]
    if not voiced_idx:
        return write_tone_placeholder(out_wav, seconds=max(0.6, target_seconds), sr=sr)

    rng = random.Random(len(y))
    chunks = []; total = 0
    while total < int(target_seconds * sr):
        i = rng.choice(voiced_idx)
        s = max(0, i * hop - rng.randint(0, int(0.01 * sr)))
        e = min(len(y), s + rng.randint(int(0.06 * sr), int(0.16 * sr)))
        seg = y[s:e].copy()
        seg = apply_fade(seg, sr, fade_ms=10)
        chunks.append(seg); total += len(seg)
        if len(chunks) > 1000: break
    out = concat_crossfade(chunks, sr, xfade_ms=10)
    try:
        import scipy.signal as ss
        b, a = ss.butter(4, [70 / (sr / 2), 7000 / (sr / 2)], btype="band")
        out = ss.lfilter(b, a, out).astype(np.float32)
    except Exception:
        pass
    sf.write(out_wav, out, sr)
    return out_wav


# ========= 主流程 =========
def main():
    ap = argparse.ArgumentParser()
    #ap.add_argument("--tasks_csv", default="aigc_speech_generation_tasks/aigc_speech_generation_tasks.csv")
    ap.add_argument("--tasks_csv", default="tests/aigc_speech_generation_tasks.csv")
    ap.add_argument("--team_name", default="result")
    ap.add_argument("--limit", type=int, default=0)

    # backends
    ap.add_argument("--idx_model_dir", default="checkpoints")
    ap.add_argument("--idx_cfg", default="checkpoints/config.yaml")
    ap.add_argument("--k_idx", type=int, default=2)

    ap.add_argument("--use_xtts_if_cached", action="store_true")
    ap.add_argument("--k_xtts", type=int, default=0)

    ap.add_argument("--k_f5", type=int, default=0)

    ap.add_argument("--use_piper_if_available", action="store_true")
    ap.add_argument("--piper_model", default="models/piper/zh_CN-huayan-medium.onnx")  # models/piper/zh_CN-huayan-medium.onnx

    # # ASR/scoring without aiss
    # ap.add_argument("--no_asr", action="store_true")
    # ap.add_argument("--asr_backend", choices=["faster", "openai"], default="faster")
    # ap.add_argument("--asr_device", choices=["auto", "cpu", "cuda"], default="auto")
    # ap.add_argument("--asr_model", default="large-v3")
    # ap.add_argument("--asr_download_root", default="models/whisper-large-v3")
    # ap.add_argument("--asr_compute_type", default="float16")

    #ap.add_argument("--w_wer", type=float, default=0.4)
    # ap.add_argument("--w_sim", type=float, default=0.4)
    # ap.add_argument("--score_log", default="logs/score_log.csv")

    # --- ADD: argparse options ---
    ap.add_argument("--no_asr", action="store_true", help="Disable ASR/WER scoring.")
    ap.add_argument("--asr_backend", type=str, default="openai", choices=["faster", "openai"])
    ap.add_argument("--asr_model", type=str, default="large-v3")
    ap.add_argument("--asr_device", type=str, default="cpu")
    ap.add_argument("--asr_compute_type", type=str, default="float16")
    ap.add_argument("--asr_download_root", type=str, default="models/whisper-large-v3")

    # AASIST 相关（可选）
    ap.add_argument("--use_aasist", action="store_true", help="Enable AASIST bonafide scoring.")
    ap.add_argument("--w_aasist", type=float, default=0.2, help="Weight for AASIST in final score.")

    ap.add_argument("--score_log", type=str, default="logs/score_log.csv", help="Per-candidate score log CSV path.")
    # 现有权重若未暴露，也可补充（若已有就忽略）
    ap.add_argument("--w_wer", type=float, default=0.4)
    ap.add_argument("--w_sim", type=float, default=0.4)


    # 双路参考预处理
    ap.add_argument("--ref_sr", type=int, default=16000)
    ap.add_argument("--ref_pick_sec", type=float, default=6.0)
    ap.add_argument("--ref_max_in_sec", type=float, default=60.0)
    ap.add_argument("--ref_vad_aggr", type=int, default=2)
    ap.add_argument("--ref_highpass", type=int, default=60)
    ap.add_argument("--ref_lowpass", type=int, default=8000)
    ap.add_argument("--ref_hum_notch", type=int, default=0)  # 0=auto, 50 or 60 to force
    ap.add_argument("--ref_embed_rms", type=float, default=-23.0)
    ap.add_argument("--ref_asr_rms", type=float, default=-23.0)

    args = ap.parse_args()


    df = pd.read_csv(args.tasks_csv)
    if args.limit > 0:
        df = df.head(args.limit)

    out_dir = Path(args.team_name if args.team_name.endswith("-result") or args.team_name == "result"
                   else f"{args.team_name}-result")
    out_dir.mkdir(parents=True, exist_ok=True)

    done_map = list_official_wavs(out_dir)
    done_set = set(done_map.keys())
    print(f"[resume] Found {len(done_set)} finished wav(s) in {out_dir}/ (skipping them).")

    # 评分器（外置模块）
    # scorer = CandidateScorer(
    #     asr_backend=args.asr_backend,
    #     asr_model=args.asr_model,
    #     asr_device=args.asr_device,
    #     asr_compute_type=args.asr_compute_type,
    #     asr_download_root=args.asr_download_root,
    #     no_asr=args.no_asr,
    #     w_wer=args.w_wer,
    #     w_sim=args.w_sim,
    #     log_path=args.score_log,
    # )

    # --- ADD: init scorer once ---
    scorer = CandidateScorer(
        asr_backend=args.asr_backend,
        asr_model=args.asr_model,
        asr_device=args.asr_device,
        asr_compute_type=args.asr_compute_type,
        asr_download_root=args.asr_download_root,
        no_asr=args.no_asr,
        w_wer=args.w_wer,
        w_sim=args.w_sim,
        use_aasist=args.use_aasist,
        w_aasist=args.w_aasist,
        aasist_device=args.asr_device,       # 直接复用 asr_device
        log_path=args.score_log,
        echo = True,
    )
    print(f"[runner] score log -> {Path(args.score_log).resolve()}")


    # 初始化合成后端
    idx = None
    try:
        idx = IndexTTSBackend(model_dir=args.idx_model_dir, cfg_path=args.idx_cfg)
    except Exception as e:
        print("IndexTTS init failed, skip:", e)

    xtts = None
    if args.use_xtts_if_cached and xtts_available():
        try:
            xtts = XTTSBackend()
            print("XTTS enabled by local cache.")
        except Exception as e:
            print("XTTS init failed, skip:", e)

    f5 = None
    try:
        f5 = F5TTSCLIBackend()
    except Exception as e:
        print("F5TTS init failed, skip:", e)

    piper = None
    if args.use_piper_if_available and piper_available(args.piper_model):
        try:
            piper = PiperBackend(args.piper_model)
            print("Piper fallback enabled.")
        except Exception as e:
            print("Piper init failed, skip:", e)

    # 参考预处理配置
    # ref_cfg = DualRefCfg(
    #     target_sr=args.ref_sr,
    #     #pick_seconds=args.ref_pick_sec,
    #     max_in_seconds=args.ref_max_in_sec,
    #     vad_aggr=args.ref_vad_aggr,
    #     highpass_hz=args.ref_highpass,
    #     lowpass_hz=args.ref_lowpass,
    #     hum_notch_hz=None if args.ref_hum_notch == 0 else int(args.ref_hum_notch),
    #     rms_dbfs_embed=args.ref_embed_rms,
    #     rms_dbfs_asr=args.ref_asr_rms

    #     pick_seconds_embed=8.0,   # 音色参考 8 秒（更保真）
    #     pick_seconds_asr=60.0,    # ASR 参考 20 秒（更稳的断句/可懂性）
    # #vad_aggr=2,               # 噪声很大可调 3
    # )

    ref_cfg = DualRefCfg(
        target_sr=args.ref_sr,
        max_in_seconds=args.ref_max_in_sec,
        vad_aggr=1,                # embed 更保守：1
        vad_pad_ms=300,
        highpass_hz=50,            # 适度高通即可
        lowpass_hz=None,           # embed 不做低通（尽量保高频细节）
        hum_notch_hz=None if args.ref_hum_notch == 0 else int(args.ref_hum_notch),

        # —— embed（音色）路：尽量不去噪，不做 RMS 归一 —— 
        snr_mild_th=18.0,                  # 更高阈值：大多数样本不去噪
        snr_bad_th=10.0,
        denoise_strength_embed_mild=0.15,  # 再轻一些
        denoise_strength_embed_light=0.0,  # 几乎不去噪
        rms_dbfs_embed=None,               # 不做 RMS 归一
        pick_seconds_embed=8.0,            # 参考拉到 8s（或 10–12s）

        # —— asr（可懂性）路：更重的清洁 —— 
        denoise_strength_asr=0.9,
        rms_dbfs_asr=-23.0,
        pick_seconds_asr=12.0,             # ASR 更长（断句/可懂性更稳）
    )


    REF_DIR = Path("aigc_speech_generation_tasks")
    errors = []; used_collage = used_tone = used_piper = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Synthesize"):
        utt = int(row["utt"])
        if utt in done_set:
            continue

        raw_text = str(row["text"])
        text = normalize_text(raw_text)
        ref_in = REF_DIR / str(row["reference_speech"])
        out_final = out_dir / f"{utt}.wav"
        if out_final.exists():
            continue

        # 参考缺失 → 兜底
        if not ref_in.exists():
            if piper is not None:
                try:
                    piper.synth_sentencewise(text, out_final); used_piper += 1
                    continue
                except Exception:
                    pass
            try:
                write_tone_placeholder(out_final, seconds=estimate_duration(text)); used_tone += 1
            except Exception:
                pass
            continue

        # 截前 60 秒（或自定义）
        ref_cut = Path(cut_reference(ref_in, max_sec=int(args.ref_max_in_sec), cache_dir=".cache/ref_trim"))

        # 双路参考（embed_ref 用于音色，相似度；asr_ref 供你有需要时用于 ASR 流）
        ref_cache_dir = os.path.join(".cache", "refproc", str(utt))
        embed_ref, asr_ref = preprocess_reference_dual(str(ref_cut), ref_cache_dir, cfg=ref_cfg)

        # === 合成并收集候选（记录 tag 以便日志） ===
        cand_pairs: list[tuple[Path, str]] = []

        if idx is not None:
            for seed in range(args.k_idx):
                tmp = out_dir / f"_tmp_{utt}_idx_{seed}.wav"
                try:
                    idx.synth_sentencewise(Path(embed_ref), text, tmp, seed=seed)
                    cand_pairs.append((tmp, f"idx_{seed}"))
                except Exception as e:
                    errors.append((utt, "IndexTTS", repr(e)))

        if xtts is not None:
            for seed in range(args.k_xtts):
                tmp = out_dir / f"_tmp_{utt}_xtts_{seed}.wav"
                try:
                    xtts.synth_sentencewise(Path(embed_ref), text, tmp, seed=seed)
                    cand_pairs.append((tmp, f"xtts_{seed}"))
                except Exception as e:
                    errors.append((utt, "XTTS", repr(e)))

        if f5 is not None:
            for k in range(args.k_f5):
                tmp = out_dir / f"_tmp_{utt}_f5_{k}.wav"
                try:
                    f5.synth_sentencewise(Path(embed_ref), text, tmp)
                    cand_pairs.append((tmp, f"f5_{k}"))
                except Exception as e:
                    errors.append((utt, "F5TTS", repr(e)))

        # 候选为空 → Piper/拼贴/哔声
        if not cand_pairs:
            if piper is not None:
                try:
                    piper.synth_sentencewise(text, out_final); used_piper += 1
                    continue
                except Exception:
                    pass
            try:
                ref_mimic_collage(Path(embed_ref), estimate_duration(text), out_final); used_collage += 1
            except Exception:
                write_tone_placeholder(out_final, seconds=estimate_duration(text)); used_tone += 1
            continue

        # === 打分选优（写入日志） ===
        try:
            best, info = scorer.score_candidates(Path(embed_ref), text, cand_pairs, utt)
        except Exception as e:
            errors.append((utt, "Scoring", repr(e)))
            best = cand_pairs[0][0]

        # 收尾：移动最佳，清理其余
        shutil.move(str(best), str(out_final))
        for w, _tag in cand_pairs:
            if w.exists():
                try: w.unlink()
                except: pass

        # 轻 dither + fade
        try:
            y, sr = sf.read(out_final)
            noise = (np.random.randn(len(y)) * 1e-4).astype(np.float32)
            y = np.clip(y.astype(np.float32) + noise, -1.0, 1.0)
            y = apply_fade(y, sr, fade_ms=10)
            sf.write(out_final, y, sr)
        except Exception:
            pass

        gc.collect()

    # 校验并补齐坏文件
    full = pd.read_csv(args.tasks_csv)
    for _, row in full.iterrows():
        utt = int(row["utt"])
        wav_path = out_dir / f"{utt}.wav"
        if not wav_path.exists() or wav_path.stat().st_size <= 2048:
            try:
                rp_in = REF_DIR / str(row["reference_speech"])
                rp_cut = Path(cut_reference(rp_in, max_sec=int(args.ref_max_in_sec), cache_dir=".cache/ref_trim"))
                ref_cache_dir = os.path.join(".cache", "refproc", str(utt))
                embed_ref, _ = preprocess_reference_dual(str(rp_cut), ref_cache_dir, cfg=ref_cfg)
                if piper is not None:
                    piper.synth_sentencewise(str(row["text"]), wav_path); used_piper += 1
                else:
                    ref_mimic_collage(Path(embed_ref), estimate_duration(str(row["text"])), wav_path); used_collage += 1
            except Exception:
                write_tone_placeholder(wav_path, seconds=estimate_duration(str(row["text"]))); used_tone += 1

    # 导出 CSV & ZIP
    full["synthesized_speech"] = [f"{i}.wav" for i in full["utt"]]
    csv_out = out_dir / f"{out_dir.name}.csv"
    full.to_csv(csv_out, index=False)

    zip_path = Path(f"{out_dir.name}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in out_dir.rglob("*"):
            zf.write(p, p.relative_to(out_dir.parent))

    print(f"\nDone. Dir: {out_dir}  Zip: {zip_path}")
    if errors:
        print(f"Failures: {len(errors)} (first 5)")
        for e in errors[:5]: print("  ", e)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", str(Path("./.hf_cache").resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path("./.cache").resolve()))
    main()
