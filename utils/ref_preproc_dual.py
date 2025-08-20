# utils/ref_preproc_dual.py
# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfilt, iirnotch, lfilter

# 可选：更强谱减
try:
    import noisereduce as nr
    _HAS_NR = True
except Exception:
    _HAS_NR = False

# WebRTC VAD
try:
    import webrtcvad
    _HAS_VAD = True
except Exception:
    _HAS_VAD = False


@dataclass
class DualRefCfg:
    # 采样与输入限制
    target_sr: int = 16000
    max_in_seconds: float = 60.0  # 原始参考最长读取时长

    # 参考目标时长（👉 双路独立）
    pick_seconds_embed: float = 6.0   # 声纹/相似度用的参考时长（轻处理）
    pick_seconds_asr: float = 6.0     # ASR/断句用的参考时长（重处理）

    # 端点淡入淡出
    fade_ms: int = 12

    # 滤波
    highpass_hz: Optional[int] = 60
    lowpass_hz: Optional[int] = 8000
    hum_notch_hz: Optional[int] = None  # 50 或 60（None=自动判断）

    # 归一化
    peak_dbfs: float = -3.0
    rms_dbfs_embed: Optional[float] = -23.0
    rms_dbfs_asr: Optional[float] = -23.0

    # VAD
    use_vad: bool = True
    vad_aggr: int = 2
    vad_frame_ms: int = 20
    vad_pad_ms: int = 200

    # 动态去噪阈值（基于 SNR/HNR）
    snr_mild_th: float = 15.0
    snr_bad_th: float = 8.0
    denoise_strength_embed_mild: float = 0.4
    denoise_strength_embed_light: float = 0.2
    denoise_strength_asr: float = 0.9

    # 选段策略
    prefer_harmonic: bool = True
    crossfade_ms: float = 12.0


# def _ensure_mono(y: np.ndarray) -> np.ndarray:
#     return y.mean(axis=0) if y.ndim == 2 else y

def _ensure_mono_float32(y: np.ndarray) -> np.ndarray:
    if y.ndim == 2:
        y = y.mean(axis=0)
    if y.dtype != np.float32:
        y = y.astype(np.float32, copy=False)
    # 避免 NaN/Inf
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return y

def _to_float32(y: np.ndarray) -> np.ndarray:
    """把各种整型/浮点统一成 [-1,1] 的 float32。"""
    if np.issubdtype(y.dtype, np.floating):
        y = y.astype(np.float32)
    else:
        # 有符号整型 → 归一化到 [-1,1]
        info = np.iinfo(y.dtype)
        y = y.astype(np.float32) / max(abs(info.min), info.max)
    # 极端保护：防止超界
    m = np.max(np.abs(y)) + 1e-12
    if m > 1.0:
        y = y / m
    return y


def _load_audio(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=False)
    y = _ensure_mono_float32(y)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, fix=True)
        sr = target_sr
    return y, sr


def _peak_norm(y: np.ndarray, peak_dbfs: float) -> np.ndarray:
    y = y.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(y))) + 1e-12
    target = 10 ** (peak_dbfs / 20.0)
    if peak > target:
        y = y * (target / peak)
    return y


def _rms_norm(y: np.ndarray, target_dbfs: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(y ** 2))) + 1e-12
    target = 10 ** (target_dbfs / 20.0)
    return (y * (target / rms)).astype(np.float32)


def _butter_sos(kind: str, cutoff_hz: float, sr: int, order: int = 4):
    nyq = sr * 0.5
    norm = cutoff_hz / nyq
    return butter(order, norm, btype=("highpass" if kind == "high" else "lowpass"), output="sos")


def _apply_filters(y: np.ndarray, sr: int, hp: Optional[int], lp: Optional[int]) -> np.ndarray:
    y = y.astype(np.float32)
    if hp:
        y = sosfilt(_butter_sos("high", hp, sr), y)
    if lp and lp < int(sr * 0.49):
        y = sosfilt(_butter_sos("low", lp, sr), y)
    return y.astype(np.float32)


def _apply_notch(y: np.ndarray, sr: int, freq: int) -> np.ndarray:
    # 陷波 50/60Hz（Q=30）
    b, a = iirnotch(w0=freq/(sr/2), Q=30)
    return lfilter(b, a, y).astype(np.float32)


def _simple_denoise(y: np.ndarray, sr: int, strength: float) -> np.ndarray:
    if _HAS_NR:
        try:
            return nr.reduce_noise(y=y.astype(np.float32), sr=sr, stationary=False)
        except Exception:
            pass
    # 轻量谱减（strength 调制）
    y = y.astype(np.float32)
    stft = librosa.stft(y, n_fft=1024, hop_length=256)
    S = np.abs(stft) + 1e-8
    noise = np.quantile(S, 0.10, axis=1, keepdims=True)
    Sd = np.maximum(S - strength * noise, 0.0)
    yd = librosa.istft(Sd * np.exp(1j * np.angle(stft)), hop_length=256, length=len(y))
    return yd.astype(np.float32)


def _bytes_frames(y16: bytes, sr: int, frame_ms: int) -> List[bytes]:
    n = int(sr * frame_ms / 1000) * 2
    return [y16[i:i+n] for i in range(0, len(y16), n)]


def _vad_segments(y: np.ndarray, sr: int, aggr: int, frame_ms: int, pad_ms: int) -> List[Tuple[int,int]]:
    assert _HAS_VAD, "webrtcvad 未安装"
    s16 = (y * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
    frames = _bytes_frames(s16, sr, frame_ms)
    hop = int(sr * frame_ms / 1000)
    pad = int(sr * pad_ms / 1000)

    segs = []
    cur = None
    vad = webrtcvad.Vad(aggr)
    for i, f in enumerate(frames):
        if len(f) != len(frames[0]):
            break
        try:
            sp = vad.is_speech(f, sr)
        except Exception:
            sp = False
        if sp and cur is None:
            cur = [i*hop, i*hop+hop]
        elif sp and cur is not None:
            cur[1] = i*hop+hop
        elif (not sp) and cur is not None:
            segs.append(tuple(cur)); cur = None
    if cur is not None:
        segs.append(tuple(cur))

    # 合并 + pad
    merged = []
    for s,e in segs:
        s = max(0, s-pad); e = min(len(y), e+pad)
        if not merged or s > merged[-1][1] + pad:
            merged.append([s,e])
        else:
            merged[-1][1] = e
    return [(s,e) for s,e in merged]


def _fade_io(y: np.ndarray, sr: int, ms: int) -> np.ndarray:
    n = min(int(sr*ms/1000), len(y)//2)
    if n <= 0:
        return y
    fade_in = np.linspace(0,1,n, dtype=np.float32)
    fade_out= np.linspace(1,0,n, dtype=np.float32)
    y = y.copy()
    y[:n] *= fade_in
    y[-n:] *= fade_out
    return y.astype(np.float32)


def _energy(x: np.ndarray) -> float:
    return float(np.mean(x.astype(np.float32) ** 2))


def _hnr_score(x: np.ndarray, sr: int) -> float:
    """
    粗略 HNR：Harmonic/Percussive after HPSS on STFT magnitude
    保证输入为 float32；段太短时回退。
    """
    x = x.astype(np.float32, copy=False)
    if len(x) < 1024:  # 极短时直接回退
        return 1.0
    try:
        X = librosa.stft(x, n_fft=1024, hop_length=256)
        H, P = librosa.effects.hpss(np.abs(X))
        h = np.mean(np.abs(H))
        p = np.mean(np.abs(P)) + 1e-8
        return float(h / p)
    except Exception:
        return 1.0


def _pick_segments(y: np.ndarray, sr: int, segs: List[Tuple[int,int]],
                   want_sec: float, prefer_harmonic: bool, crossfade_ms: float) -> np.ndarray:
    """按评分挑选若干段拼接到目标时长（不够就用现有长度，不报错）"""
    y = y.astype(np.float32)
    if want_sec <= 0:
        return np.zeros(0, dtype=np.float32)

    if not segs:
        take = min(len(y), int(want_sec*sr))
        return y[:take]

    scored = []
    for s,e in segs:
        seg = y[s:e].astype(np.float32)
        if len(seg) < int(0.1*sr):  # 極短片段跳過
            continue
        if prefer_harmonic:
            score = 0.7 * _hnr_score(seg, sr) + 0.3 * _energy(seg)
        else:
            score = _energy(seg)
        scored.append((score, s, e))
    if not scored:
        take = min(len(y), int(want_sec*sr))
        return y[:take]

    scored.sort(key=lambda t: t[0], reverse=True)

    need = int(want_sec * sr)
    got = 0
    out = []
    xf = int(sr * crossfade_ms / 1000)
    for _, s, e in scored:
        if got >= need:
            break
        seg = y[s:e]
        take = min(len(seg), need - got)
        seg = seg[:take]
        if not out:
            out.append(seg)
        else:
            a = out[-1]
            if xf > 0 and len(a) > xf and len(seg) > xf:
                fo = np.linspace(1,0,xf, dtype=np.float32)
                fi = np.linspace(0,1,xf, dtype=np.float32)
                a[-xf:] = a[-xf:] * fo + seg[:xf] * fi
                out[-1] = a
                out.append(seg[xf:])
            else:
                out.append(seg)
        got += take

    return np.concatenate(out).astype(np.float32) if out else y[:need]


def _snr_estimate(y: np.ndarray, sr: int, segs: List[Tuple[int,int]]) -> float:
    """基于 VAD：speech 段能量 / 非语音段能量（dB）"""
    y = y.astype(np.float32)
    mask = np.zeros(len(y), dtype=bool)
    for s,e in segs:
        mask[s:e] = True
    speech = y[mask]
    noise = y[~mask]
    if len(speech) < sr*0.2 or len(noise) < sr*0.2:
        # fallback：分位数估计
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
        sp = np.quantile(S, 0.80); ns = np.quantile(S, 0.20) + 1e-8
        return float(20*np.log10(sp/ns))
    ps = np.mean(speech**2) + 1e-12
    pn = np.mean(noise**2) + 1e-12
    return float(10*np.log10(ps/pn))


def _auto_detect_hum(y: np.ndarray, sr: int) -> Optional[int]:
    """简易 50/60Hz 嗡声检测：哪个峰更高返回哪个（弱则不做）。"""
    y = y.astype(np.float32)
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024)).mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

    def band_power(f0, bw=3):
        idx = np.where((freqs >= f0-bw) & (freqs <= f0+bw))[0]
        return float(S[idx].mean()) if len(idx) > 0 else 0.0

    p50, p60 = band_power(50), band_power(60)
    if max(p50, p60) < 0.001:
        return None
    return 50 if p50 >= p60 else 60


def preprocess_reference_dual(
    in_path: str,
    out_dir: str,
    cfg: DualRefCfg = DualRefCfg()
) -> Tuple[str, str]:
    """
    读取参考音频 → 轻/重两条参考：
      - embed_ref.wav（轻处理，供音色/相似度抽取；目标长度 cfg.pick_seconds_embed）
      - asr_ref.wav   （重处理，供ASR/对齐/可懂性；目标长度 cfg.pick_seconds_asr）
    返回 (embed_ref_path, asr_ref_path)
    """
    os.makedirs(out_dir, exist_ok=True)
    y, sr = _load_audio(in_path, cfg.target_sr)

    # 限时
    if cfg.max_in_seconds and len(y) > cfg.max_in_seconds*sr:
        y = y[:int(cfg.max_in_seconds*sr)]

    # 初步滤波（轻）
    y0 = _apply_filters(y, sr, cfg.highpass_hz, cfg.lowpass_hz)

    # VAD
    if cfg.use_vad and _HAS_VAD:
        segs = _vad_segments(y0, sr, cfg.vad_aggr, cfg.vad_frame_ms, cfg.vad_pad_ms)
    else:
        segs = [(0, len(y0))]

    # SNR 评估（用于去噪决策）
    snr = _snr_estimate(y0, sr, segs)

    # 自动 50/60Hz 嗡声陷波（如果明显）
    hum = cfg.hum_notch_hz or _auto_detect_hum(y0, sr)
    if hum in (50, 60):
        y0 = _apply_notch(y0, sr, hum)

    # —— 声纹参考（轻处理）——
    y_embed = _pick_segments(y0, sr, segs, cfg.pick_seconds_embed, cfg.prefer_harmonic, cfg.crossfade_ms)
    if snr < cfg.snr_mild_th:
        strength = cfg.denoise_strength_embed_light if snr < cfg.snr_bad_th else cfg.denoise_strength_embed_mild
        y_embed = _simple_denoise(y_embed, sr, strength=strength)
    y_embed = _fade_io(y_embed, sr, cfg.fade_ms)
    y_embed = _peak_norm(y_embed, cfg.peak_dbfs)
    if cfg.rms_dbfs_embed is not None:
        y_embed = _rms_norm(y_embed, cfg.rms_dbfs_embed)
    embed_path = os.path.join(out_dir, "embed_ref.wav")
    sf.write(embed_path, y_embed, sr)

    # —— ASR 参考（重处理）——
    y_asr = _pick_segments(y0, sr, segs, cfg.pick_seconds_asr, cfg.prefer_harmonic, cfg.crossfade_ms)
    y_asr = _simple_denoise(y_asr, sr, strength=cfg.denoise_strength_asr)
    y_asr = _apply_filters(y_asr, sr, cfg.highpass_hz or 60, min(cfg.lowpass_hz or 8000, 7500))
    y_asr = _fade_io(y_asr, sr, cfg.fade_ms)
    y_asr = _peak_norm(y_asr, cfg.peak_dbfs)
    if cfg.rms_dbfs_asr is not None:
        y_asr = _rms_norm(y_asr, cfg.rms_dbfs_asr)
    asr_path = os.path.join(out_dir, "asr_ref.wav")
    sf.write(asr_path, y_asr, sr)

    return embed_path, asr_path
