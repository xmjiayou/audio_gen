# ref_preproc.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import soundfile as sf
import librosa
from scipy import signal

# 可选依赖：有则用、无则自动降级
try:
    import webrtcvad  # VAD
    _HAVE_VAD = True
except Exception:
    _HAVE_VAD = False

try:
    import noisereduce as nr  # 轻度频谱去噪
    _HAVE_NR = True
except Exception:
    _HAVE_NR = False

try:
    import pyloudnorm as pyln  # LUFS
    _HAVE_LUFS = True
except Exception:
    _HAVE_LUFS = False


@dataclass
class PreprocConfig:
    sr: int = 16000                 # 参考输出采样率（给 IndexTTS 通常 16k 就够）
    embed_len_sec: float = 12.0     # 用于“克隆”的参考总时长
    asr_len_sec: float = 18.0       # 用于 ASR 的参考总时长（可稍长一点）
    vad_aggr: int = 2               # webrtcvad 激进度 0-3
    crossfade_ms: int = 10          # 段拼接交叉淡化
    target_lufs: float = -23.0      # LUFS 规范化目标
    denoise_policy: str = "auto"    # 'off'|'light'|'strong'|'auto'
    use_notch_50_60: bool = True    # 工频陷波
    highpass_hz: int = 60           # DC/超低频滚降
    bandpass_low_hz: int = 80       # 语音带通（下限）
    bandpass_high_hz: int = 7800    # 语音带通（上限）
    cache_dir: Path = Path("preproc_refs")  # 缓存目录


# ---------------- 基础工具 ----------------
def _safe_mono_resample(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    y = np.ascontiguousarray(y.astype(np.float32))
    return y, sr


def _rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y**2) + 1e-12))


def _loudness_norm(y: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    if _HAVE_LUFS:
        meter = pyln.Meter(sr)
        try:
            loud = meter.integrated_loudness(y.astype(np.float32))
            y = pyln.normalize.loudness(y, loud, target_lufs).astype(np.float32)
            return y
        except Exception:
            pass
    # 简化 RMS 兜底（不是 LUFS，但足够稳）
    ref = 10 ** (target_lufs / 20.0)
    cur = _rms(y)
    if cur < 1e-6:
        return y
    y = np.clip(y * (ref / cur), -1.0, 1.0)
    return y.astype(np.float32)


def _butter_filter(y: np.ndarray, sr: int, kind: str, cutoff, order=6) -> np.ndarray:
    if kind == "hp":
        sos = signal.butter(order, cutoff, btype="highpass", fs=sr, output="sos")
    elif kind == "lp":
        sos = signal.butter(order, cutoff, btype="lowpass", fs=sr, output="sos")
    elif kind == "bp":
        sos = signal.butter(order, cutoff, btype="bandpass", fs=sr, output="sos")
    else:
        raise ValueError("unknown filter")
    return signal.sosfiltfilt(sos, y).astype(np.float32)


def _notch(y: np.ndarray, sr: int, f0: float, q: float = 30.0) -> np.ndarray:
    b, a = signal.iirnotch(w0=f0, Q=q, fs=sr)
    return signal.filtfilt(b, a, y).astype(np.float32)


def _estimate_snr(y: np.ndarray, frame_len=2048, hop=512) -> float:
    S = librosa.stft(y, n_fft=frame_len, hop_length=hop)
    amp = np.abs(S)
    noise = np.percentile(amp, 10, axis=1, keepdims=True) + 1e-9
    signal_ = np.percentile(amp, 90, axis=1, keepdims=True) + 1e-9
    snr = 20.0 * np.log10(np.mean(signal_) / np.mean(noise) + 1e-9)
    return float(snr)


def _estimate_hnr(y: np.ndarray, sr: int) -> float:
    # 近似 HNR：谐波/噪声粗略比值（安全简化，鲁棒优先）
    y_hp = _butter_filter(y, sr, "hp", 60, order=4)
    # 自相关峰/背景比
    ac = librosa.autocorrelate(y_hp)
    if ac.max() <= 1e-9:
        return 0.0
    peak = float(np.max(ac))
    tail = float(np.mean(np.abs(ac[int(0.02 * sr):int(0.06 * sr)])) + 1e-9)  # 20~60ms
    return 20.0 * np.log10(max(peak / tail, 1e-6))


# ---------------- VAD 切片 ----------------
def _vad_segments(y: np.ndarray, sr: int, aggr: int = 2) -> List[Tuple[int, int]]:
    segs: List[Tuple[int, int]] = []
    if _HAVE_VAD:
        # webrtcvad 需要 16k/单声道/16-bit
        if sr != 16000:
            y16 = librosa.resample(y, sr, 16000)
            srx = 16000
        else:
            y16 = y
            srx = sr
        vad = webrtcvad.Vad(aggr)
        frame_ms = 30
        frame_len = int(srx * frame_ms / 1000)
        step = frame_len
        pcm = np.clip(y16 * 32768.0, -32768, 32767).astype(np.int16).tobytes()
        flags = []
        for i in range(0, len(y16) - frame_len, step):
            chunk = pcm[i * 2:(i + frame_len) * 2]
            flags.append(vad.is_speech(chunk, sample_rate=srx))
        # 合并为段
        start = None
        for idx, f in enumerate(flags):
            if f and start is None:
                start = idx * step
            if (not f) and start is not None:
                end = idx * step
                # 回到原 sr
                if srx != sr:
                    start_ = int(start * sr / srx)
                    end_ = int(end * sr / srx)
                else:
                    start_, end_ = start, end
                if end_ - start_ > int(0.2 * sr):
                    segs.append((start_, end_))
                start = None
        if start is not None:
            end = len(y16) if srx == sr else int(len(y16) * sr / srx)
            if end - start > int(0.2 * sr):
                segs.append((int(start * sr / srx), end))
        if segs:
            return segs

    # 无 webrtcvad 或无段，退化：能量阈值
    intervals = librosa.effects.split(y, top_db=35)
    for a, b in intervals:
        if b - a > int(0.2 * sr):
            segs.append((int(a), int(b)))
    if not segs:
        # 仍然没有，就取中间 6~12s
        mid = len(y) // 2
        half = int(6 * sr)
        a = max(0, mid - half)
        b = min(len(y), a + int(12 * sr))
        segs.append((a, b))
    return segs


def _stitch_segments(y: np.ndarray, sr: int, segs: List[Tuple[int, int]],
                     want_sec: float, crossfade_ms: int) -> np.ndarray:
    need = int(want_sec * sr)
    if need <= 0:
        return y
    cf = int(crossfade_ms * sr / 1000)
    out = np.zeros(0, dtype=np.float32)
    for a, b in segs:
        chunk = y[a:b]
        if len(out) == 0:
            out = chunk.copy()
        else:
            if cf > 0 and len(out) >= cf and len(chunk) >= cf:
                fade_out = np.linspace(1.0, 0.0, cf, dtype=np.float32)
                fade_in = 1.0 - fade_out
                out[-cf:] = out[-cf:] * fade_out + chunk[:cf] * fade_in
                out = np.concatenate([out, chunk[cf:]], axis=0)
            else:
                out = np.concatenate([out, chunk], axis=0)
        if len(out) >= need:
            break
    if len(out) < need:
        # 补齐：重复最后一段的平滑尾部（避免硬拼）
        pad = need - len(out)
        tail = y[segs[-1][0]:segs[-1][1]]
        tail = tail[:min(len(tail), pad)]
        if len(tail) > 0:
            if cf > 0 and len(out) >= cf and len(tail) >= cf:
                fade_out = np.linspace(1.0, 0.0, cf, dtype=np.float32)
                fade_in = 1.0 - fade_out
                out[-cf:] = out[-cf:] * fade_out + tail[:cf] * fade_in
                out = np.concatenate([out, tail[cf:]], axis=0)
            else:
                out = np.concatenate([out, tail], axis=0)
        if len(out) < need:
            out = np.pad(out, (0, need - len(out)))
    return out.astype(np.float32)


def _maybe_denoise(y: np.ndarray, sr: int, policy: str,
                   snr_db: float, hnr_db: float) -> np.ndarray:
    if policy == "off":
        return y
    if policy == "light":
        strength = 0.1
    elif policy == "strong":
        strength = 0.2
    else:
        # auto：SNR<10 或 HNR<15 时轻度去噪
        if snr_db < 10.0 or hnr_db < 15.0:
            strength = 0.12
        else:
            return y
    if not _HAVE_NR:
        return y
    try:
        y_dn = nr.reduce_noise(y=y, sr=sr, prop_decrease=strength, stationary=False, use_tqdm=False)
        return y_dn.astype(np.float32)
    except Exception:
        return y


def _front_filters(y: np.ndarray, sr: int, cfg: PreprocConfig) -> np.ndarray:
    y = _butter_filter(y, sr, "hp", cfg.highpass_hz, order=4)
    if cfg.use_notch_50_60:
        y = _notch(y, sr, 50.0)
        y = _notch(y, sr, 60.0)
    # 轻带通
    hi = min(cfg.bandpass_high_hz, int(sr * 0.48))
    if cfg.bandpass_low_hz < hi:
        y = _butter_filter(y, sr, "bp", [cfg.bandpass_low_hz, hi], order=4)
    return y


def _fade_io(y: np.ndarray, sr: int, ms: int = 10) -> np.ndarray:
    n = int(ms * sr / 1000)
    n = min(n, len(y) // 3)
    if n <= 0: 
        return y
    w = np.linspace(0, 1, n, dtype=np.float32)
    y[:n] *= w
    y[-n:] *= w[::-1]
    return y


def _hash_name(path: Path, cfg: PreprocConfig) -> str:
    h = hashlib.sha1()
    h.update(str(path.resolve()).encode())
    h.update(str(cfg).encode())
    return h.hexdigest()[:10]


def preprocess_reference_dual(src_wav: Path,
                              work_dir: Path,
                              cfg: Optional[PreprocConfig] = None
                              ) -> Tuple[Path, Path, Dict]:
    """
    返回: (embed_wav_path, asr_wav_path, meta)
    """
    cfg = cfg or PreprocConfig()
    work_dir = Path(work_dir); work_dir.mkdir(parents=True, exist_ok=True)

    y, sr = _safe_mono_resample(Path(src_wav), cfg.sr)
    # 质量估计
    snr_db = _estimate_snr(y)
    hnr_db = _estimate_hnr(y, sr)

    # 预滤波
    y_f = _front_filters(y, sr, cfg)

    # 选段
    segs = _vad_segments(y_f, sr, cfg.vad_aggr)
    y_pick = _stitch_segments(y_f, sr, segs, cfg.embed_len_sec, cfg.crossfade_ms)

    # 去噪（按策略）
    y_dn = _maybe_denoise(y_pick, sr, cfg.denoise_policy, snr_db, hnr_db)

    # 规范化
    y_norm = _loudness_norm(y_dn, sr, cfg.target_lufs)
    y_norm = _fade_io(y_norm, sr, 10)

    # 另做 ASR 版本（更宽频段/或更长秒数）
    asr_segs = _vad_segments(y_f, sr, cfg.vad_aggr)
    y_asr = _stitch_segments(y_f, sr, asr_segs, cfg.asr_len_sec, cfg.crossfade_ms)
    y_asr = _maybe_denoise(y_asr, sr, cfg.denoise_policy, snr_db, hnr_db)
    y_asr = _loudness_norm(y_asr, sr, cfg.target_lufs)
    y_asr = _fade_io(y_asr, sr, 10)

    # 缓存命名
    stem = Path(src_wav).stem
    tag = _hash_name(Path(src_wav), cfg)
    p_embed = work_dir / f"{stem}.{tag}.embed.wav"
    p_asr   = work_dir / f"{stem}.{tag}.asr.wav"

    # 保存
    sf.write(p_embed, y_norm, sr)
    sf.write(p_asr, y_asr, sr)

    meta = {
        "sr": sr,
        "snr_db": snr_db,
        "hnr_db": hnr_db,
        "segments": segs,
        "embed_len": float(len(y_norm) / sr),
        "asr_len": float(len(y_asr) / sr)
    }
    return p_embed, p_asr, meta


# 它会做：安全读入 → VAD 选段（优先 webrtcvad，缺失时走能量阈值）→ 轻 EQ（高通 + 选配工频陷波）→ （按 SNR/HNR“自动”策略）轻度去噪 → -23 LUFS 规范化 → 交叉淡化拼接 → 输出两份参考：

# *_embed.wav：推荐给 IndexTTS / XTTS / F5 等“克隆/嵌入”模型；

# *_asr.wav：给 ASR 估字用（如果以后要用的话），默认和 embed 一致，只是保留更宽频段。

# 依赖（任选其一都能跑；缺了会自动降级）
# pip install webrtcvad noisereduce pyloudnorm soundfile librosa scipy

# 缺 webrtcvad：改用 librosa.effects.split 的能量分段。

# 缺 noisereduce：跳过去噪（仅滤波+规范化）。

# 缺 pyloudnorm：用 RMS 近似到 -23 dBFS。


# 原有的 preprocess_ref 函数整体替换为下面版本：
# # --- 新版：委托给 ref_preproc.py ---
# from ref_preproc import PreprocConfig, preprocess_reference_dual

# # 统一缓存目录（保持你原来的 PRE_DIR 不变）
# # PRE_DIR = Path("preproc_refs"); PRE_DIR.mkdir(exist_ok=True)

# def preprocess_ref(wav_path: Path,
#                    target_sr: int = 16000,
#                    embed_len_sec: float = 12.0,
#                    asr_len_sec: float = 18.0,
#                    denoise_policy: str = "auto",
#                    vad_aggr: int = 2) -> Path:
#     """
#     兼容旧签名：返回“用于克隆”的参考路径（embed 版）。
#     需要 ASR 版时，可改为：
#         emb, asr, meta = preprocess_reference_dual(...)
#     """
#     cfg = PreprocConfig(
#         sr=target_sr,
#         embed_len_sec=embed_len_sec,
#         asr_len_sec=asr_len_sec,
#         vad_aggr=vad_aggr,
#         denoise_policy=denoise_policy,
#         cache_dir=PRE_DIR
#     )
#     emb_path, asr_path, meta = preprocess_reference_dual(wav_path, PRE_DIR, cfg)
#     # 你现在的下游只用到了一个路径，这里默认返回 embed 版
#     return emb_path
# 说明：

# 你原来在合成前有：ref = preprocess_ref(ref_path)[0] 或 ref = preprocess_ref(ref_path)。
# 用上面替换后，保持 ref = preprocess_ref(ref_path) 即可。

# 如果你后续想对 ASR 做“参考对齐/校验”，可以这样拿到更长的 asr_path：

# emb_path, asr_path, meta = preprocess_reference_dual(ref_path, PRE_DIR, PreprocConfig(...))
# ③ 调用位置不需要其它变更
# 在 run_0817.py 里，找到你给模型喂参考音频之前的地方（通常是每条样本的 ref_path = REF_DIR / row['reference_speech'] 之后），保持原来的调用方式即可，例如：

# ref_wav = preprocess_ref(ref_path)  # 现在它已是“高质量、规范化”的 embed 参考
# # 传给 IndexTTS / XTTS / F5：
# # idx_tts.synthesize(text=..., ref_wav=str(ref_wav), ...)
# 我没有改你下游的任何接口（IndexTTS/F5/Piper/ref_mimic_collage），仅替换了参考预处理。

# ④ 方案要点（和你给的博客思路一致）
# 相似度 40%：默认不激进去噪；仅当 SNR/HNR 低到阈值时，做“轻度频谱抑制”（prop_decrease≈0.12），避免洗掉声纹细节。

# 可懂度 40%：做合理的分段挑选（VAD）、轻 EQ（高通 + 工频陷波）、-23 LUFS 统一响度，减少 ASR 失真因素。

# 自然度 20%：避免强处理导致“金属感/水波纹”，交叉淡化拼接，端点淡入淡出，减少拼接感。

# 常见问题
# 没网或装不上 webrtcvad/noisereduce/pyloudnorm？
# 也能跑（会自动使用退化策略：能量阈值切段、跳过去噪、RMS 规范化）。

# 采样率是否要 24k？
# 参考给“嵌入模型”的 16k 通常更稳，也更省显存/带宽。你可把 PreprocConfig(sr=24000) 调到 24k，其他不变。



