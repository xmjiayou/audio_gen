# ref_preproc.py
# -*- coding: utf-8 -*-
import os
import io
import math
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfilt

# 可选：轻量谱减法降噪（装不上可忽略）
try:
    import noisereduce as nr
    _HAS_NR = True
except Exception:
    _HAS_NR = False

# 可选：语音端点检测
try:
    import webrtcvad
    _HAS_VAD = True
except Exception:
    _HAS_VAD = False


@dataclass
class RefPreprocCfg:
    target_sr: int = 16000
    peak_dbfs: float = -3.0            # 峰值规范化上限
    rms_dbfs: Optional[float] = -23.0  # 目标 RMS（None 则不用 RMS 归一）
    highpass_hz: Optional[int] = 50    # 低频嗡嗡/风噪，高通去掉
    lowpass_hz: Optional[int] = 8000   # 高频尖噪/嘶声，低通去掉
    do_denoise: bool = True            # 是否进行轻量降噪
    denoise_strength: float = 0.9      # 谱减强度（0~1），越高越强
    do_vad: bool = True                # 是否用 VAD 过滤出语音段
    vad_aggr: int = 2                  # WebRTC VAD 激进度 0~3（越大越严格）
    vad_frame_ms: int = 20             # VAD 帧长 10/20/30 ms
    vad_pad_ms: int = 200              # 语音段拼接上下文（防切字）
    pick_seconds: float = 6.0          # 目标参考总时长（优先挑最干净的若干段）
    max_seconds: float = 60.0          # 参考上限（原始 wav 如果太长，先裁到这个上限）
    fade_ms: int = 12                  # 端点淡入淡出，避免爆音


def _ensure_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 2:
        return np.mean(y, axis=0)
    return y

def _ensure_mono_float32(y: np.ndarray) -> np.ndarray:
    if y.ndim == 2:
        y = y.mean(axis=0)
    if y.dtype != np.float32:
        y = y.astype(np.float32, copy=False)
    # 避免 NaN/Inf
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return y

def _load_audio(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=False)
    y = _ensure_mono_float32(y.astype(np.float32))
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, fix=True)
        sr = target_sr
    return y, sr


def _peak_normalize(y: np.ndarray, peak_dbfs: float) -> np.ndarray:
    y = y.astype(np.float32, copy=False)
    peak = np.max(np.abs(y)) + 1e-9
    target = 10 ** (peak_dbfs / 20.0)
    if peak > 0:
        y = y * (target / peak) if peak > target else y
    return y


def _rms_normalize(y: np.ndarray, target_dbfs: float) -> np.ndarray:
    rms = np.sqrt(np.mean(y ** 2)) + 1e-12
    target = 10 ** (target_dbfs / 20.0)
    gain = target / rms
    return (y * gain).astype(np.float32)


def _butter_sos(kind: str, cutoff_hz: float, sr: int, order: int = 4):
    nyq = 0.5 * sr
    norm = cutoff_hz / nyq
    if kind == "high":
        return butter(order, norm, btype="highpass", output="sos")
    else:
        return butter(order, norm, btype="lowpass", output="sos")


def _apply_filters(y: np.ndarray, sr: int,
                   highpass_hz: Optional[int], lowpass_hz: Optional[int]) -> np.ndarray:
    y = y.astype(np.float32)
    if highpass_hz:
        y = sosfilt(_butter_sos("high", highpass_hz, sr), y)
    if lowpass_hz and lowpass_hz < (sr * 0.49):
        y = sosfilt(_butter_sos("low", lowpass_hz, sr), y)
    return y.astype(np.float32)


def _simple_denoise(y: np.ndarray, sr: int, strength: float = 0.9) -> np.ndarray:
    """
    纯 Python 轻量谱减（无外部模型）；若安装 noisereduce 则优先用之。
    """
    if _HAS_NR:
        try:
            return nr.reduce_noise(y=y, sr=sr)  # 自适应谱减
        except Exception:
            pass
    # 简单谱减：估计噪声谱=幅度谱的 10% 分位数
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) + 1e-8
    noise_prof = np.quantile(S, 0.10, axis=1, keepdims=True)
    S_d = np.maximum(S - strength * noise_prof, 0.0)
    phase = np.angle(librosa.stft(y, n_fft=1024, hop_length=256))
    y_d = librosa.istft(S_d * np.exp(1j * phase), hop_length=256, length=len(y))
    return y_d.astype(np.float32)


def _frame_bytes(y16: bytes, sr: int, frame_ms: int) -> List[bytes]:
    # 16-bit mono PCM 分帧
    bytes_per_frame = int(sr * frame_ms / 1000) * 2
    return [y16[i:i + bytes_per_frame] for i in range(0, len(y16), bytes_per_frame)]


def _vad_segments(y: np.ndarray, sr: int, aggr: int, frame_ms: int, pad_ms: int) -> List[Tuple[int, int]]:
    assert _HAS_VAD, "webrtcvad 未安装"
    vad = webrtcvad.Vad(aggr)
    y = (y * 32767.0).clip(-32768, 32767).astype(np.int16)
    frames = _frame_bytes(y.tobytes(), sr, frame_ms)
    hop = int(sr * frame_ms / 1000)
    pad = int(sr * pad_ms / 1000)

    voiced = []
    cur = None
    for i, f in enumerate(frames):
        is_speech = False
        if len(f) == len(frames[0]):  # 末尾零头忽略
            try:
                is_speech = vad.is_speech(f, sr)
            except Exception:
                is_speech = False
        if is_speech and cur is None:
            cur = [i * hop, i * hop + hop]
        elif is_speech and cur is not None:
            cur[1] = i * hop + hop
        elif (not is_speech) and cur is not None:
            voiced.append(tuple(cur))
            cur = None
    if cur is not None:
        voiced.append(tuple(cur))

    # pad 合并
    merged = []
    for s, e in voiced:
        s = max(0, s - pad)
        e = min(len(y), e + pad)
        if not merged or s > merged[-1][1] + pad:
            merged.append([s, e])
        else:
            merged[-1][1] = e
    return [(s, e) for s, e in merged]


def _energy(x: np.ndarray) -> float:
    return float(np.mean(x ** 2))


def _pick_top_segments(y: np.ndarray, sr: int, segs: List[Tuple[int, int]],
                       want_sec: float) -> np.ndarray:
    # 以能量优先，拼到目标时长；段间做 12ms crossfade
    y = y.astype(np.float32)
    if not segs:
        return y[: int(want_sec * sr)]

    parts = []
    scored = [(s, e, _energy(y[s:e])) for (s, e) in segs]
    scored.sort(key=lambda t: t[2], reverse=True)

    need = int(want_sec * sr)
    got = 0
    xf = int(0.012 * sr)

    for s, e, _ in scored:
        if got >= need:
            break
        seg = y[s:e]
        take = min(len(seg), need - got)
        seg = seg[:take]
        if len(parts) == 0:
            parts.append(seg)
        else:
            # crossfade
            a = parts[-1]
            if xf > 0 and len(a) > xf and len(seg) > xf:
                fade_out = np.linspace(1, 0, xf, dtype=np.float32)
                fade_in = np.linspace(0, 1, xf, dtype=np.float32)
                a[-xf:] = a[-xf:] * fade_out + seg[:xf] * fade_in
                parts[-1] = a
                parts.append(seg[xf:])
            else:
                parts.append(seg)
        got += take
    return np.concatenate(parts) if parts else y[:need]


def _fade_io(y: np.ndarray, sr: int, fade_ms: int) -> np.ndarray:
    n = int(fade_ms / 1000.0 * sr)
    n = min(n, len(y) // 2)
    if n <= 0: 
        return y
    fade_in = np.linspace(0, 1, n).astype(np.float32)
    fade_out = np.linspace(1, 0, n).astype(np.float32)
    y[:n] *= fade_in
    y[-n:] *= fade_out
    return y


def preprocess_reference(
    in_path: str,
    out_path: Optional[str] = None,
    cfg: RefPreprocCfg = RefPreprocCfg(),
) -> Tuple[np.ndarray, int, Optional[str]]:
    """
    读取参考音频 → 降噪/滤波/归一 → VAD 选段拼接 → 裁时长 → 端点淡入淡出
    返回: (波形, 采样率, 输出文件路径[如保存])
    """
    # 1) load & resample & mono
    y, sr = _load_audio(in_path, cfg.target_sr)

    # 2) 首先裁掉超长（避免后续处理过慢）
    if cfg.max_seconds and len(y) > cfg.max_seconds * sr:
        y = y[: int(cfg.max_seconds * sr)]

    # 3) 轻量降噪（可选）
    if cfg.do_denoise:
        y = _simple_denoise(y, sr, strength=cfg.denoise_strength)

    # 4) 高/低通滤波
    y = _apply_filters(y, sr, cfg.highpass_hz, cfg.lowpass_hz)

    # 5) VAD 语音段提取 + 选段
    if cfg.do_vad and _HAS_VAD:
        segs = _vad_segments(y, sr, aggr=cfg.vad_aggr,
                             frame_ms=cfg.vad_frame_ms, pad_ms=cfg.vad_pad_ms)
        y = _pick_top_segments(y, sr, segs, want_sec=cfg.pick_seconds)
    else:
        # 无 VAD 时直接截到目标秒数
        y = y[: int(cfg.pick_seconds * sr)]

    # 6) 端点淡入/淡出
    y = _fade_io(y, sr, cfg.fade_ms)

    # 7) 峰值 & RMS 归一
    y = _peak_normalize(y, cfg.peak_dbfs)
    if cfg.rms_dbfs is not None:
        y = _rms_normalize(y, cfg.rms_dbfs)

    # 8) 保存（可选）
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(out_path, y, sr)

    return y, sr, out_path
