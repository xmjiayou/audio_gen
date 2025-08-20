# utils/ref_preproc_dual.py
# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfiltfilt, sosfilt, iirnotch, filtfilt, lfilter

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
    # 基本
    target_sr: int = 16000
    max_in_seconds: float = 60.0       # 原始参考最长读取时长
    pick_seconds: float = 6.0          # 参考目标时长（最终拼出）
    fade_ms: int = 12                  # 端点淡入淡出

    # 滤波
    highpass_hz: Optional[int] = 60    # 高通去嗡/风噪
    lowpass_hz: Optional[int] = 8000   # 低通去嘶声尖噪
    hum_notch_hz: Optional[int] = None # 设为 -1 可强制关闭；None 表示自动检测 50/60

    # 归一
    peak_dbfs: float = -3.0
    rms_dbfs_embed: Optional[float] = -23.0  # 声纹参考 RMS
    rms_dbfs_asr: Optional[float] = -23.0    # ASR 参考 RMS

    # VAD
    use_vad: bool = True
    vad_aggr: int = 2                  # 0~3 越大越严格
    vad_frame_ms: int = 20
    vad_pad_ms: int = 200

    # 动态去噪决策（以 SNR/HNR 为依据）
    snr_mild_th: float = 15.0          # ≥该阈值：embed_ref 不去噪
    snr_bad_th: float = 8.0            # <该阈值：embed_ref 也只做很轻的去噪
    denoise_strength_embed_mild: float = 0.4
    denoise_strength_embed_light: float = 0.2
    denoise_strength_asr: float = 0.9  # ASR 参考用较强去噪

    # 选段策略
    prefer_harmonic: bool = True       # 依据谐波比(HNR)与能量排序
    crossfade_ms: float = 12.0


def _ensure_mono_float32(y: np.ndarray) -> np.ndarray:
    if y.ndim == 2:
        y = y.mean(axis=0)
    if y.dtype != np.float32:
        y = y.astype(np.float32, copy=False)
    # 避免 NaN/Inf
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return y


def _load_audio(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    # 注意：不指定 dtype，保持 soundfile 的默认，之后统一到 float32
    y, sr = sf.read(path, always_2d=False)
    y = _ensure_mono_float32(y)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, fix=True).astype(np.float32, copy=False)
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
    y = y.astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(y ** 2))) + 1e-12
    target = 10 ** (target_dbfs / 20.0)
    return (y * (target / rms)).astype(np.float32, copy=False)


def _butter_sos(kind: str, cutoff_hz: float, sr: int, order: int = 4):
    nyq = sr * 0.5
    norm = min(max(cutoff_hz / nyq, 1e-6), 0.999)
    btype = "highpass" if kind == "high" else "lowpass"
    return butter(order, norm, btype=btype, output="sos")


def _apply_filters(y: np.ndarray, sr: int, hp: Optional[int], lp: Optional[int]) -> np.ndarray:
    y = y.astype(np.float32, copy=False)
    if hp and hp > 0:
        sos = _butter_sos("high", hp, sr)
        # 点数太短时，sosfiltfilt 可能失败，回退 sosfilt
        try:
            y = sosfiltfilt(sos, y).astype(np.float32, copy=False)
        except Exception:
            y = sosfilt(sos, y).astype(np.float32, copy=False)
    if lp and lp > 0 and lp < int(sr * 0.49):
        sos = _butter_sos("low", lp, sr)
        try:
            y = sosfiltfilt(sos, y).astype(np.float32, copy=False)
        except Exception:
            y = sosfilt(sos, y).astype(np.float32, copy=False)
    return y


def _apply_notch(y: np.ndarray, sr: int, freq: int) -> np.ndarray:
    """
    50/60Hz 陷波。**不再使用 librosa.lfilter**，只用 scipy 的 filtfilt/lfilter。
    将频率归一化到 [0,1]（相对于 Nyquist），并在样本太短时回退到单向 lfilter。
    """
    y = y.astype(np.float32, copy=False)
    if freq <= 0 or freq >= sr / 2:
        return y
    w0 = float(freq) / (sr / 2.0)  # 归一化
    try:
        b, a = iirnotch(w0=w0, Q=30.0)
    except Exception:
        return y
    # 足够长则零相位滤波
    pad_needed = 3 * max(len(a), len(b))
    try:
        if len(y) > pad_needed:
            y_f = filtfilt(b, a, y)
        else:
            y_f = lfilter(b, a, y)
        return y_f.astype(np.float32, copy=False)
    except Exception:
        return y


def _simple_denoise(y: np.ndarray, sr: int, strength: float) -> np.ndarray:
    y = y.astype(np.float32, copy=False)
    if _HAS_NR:
        try:
            yd = nr.reduce_noise(y=y, sr=sr, stationary=False)
            return yd.astype(np.float32, copy=False)
        except Exception:
            pass
    # 轻量谱减（确保浮点）
    try:
        Y = librosa.stft(y.astype(np.float32, copy=False), n_fft=1024, hop_length=256)
        S = np.abs(Y).astype(np.float32, copy=False) + 1e-8
        noise = np.quantile(S, 0.10, axis=1, keepdims=True).astype(np.float32, copy=False)
        Sd = np.maximum(S - float(strength) * noise, 0.0).astype(np.float32, copy=False)
        phase = np.angle(Y).astype(np.float32, copy=False)
        yd = librosa.istft((Sd * np.exp(1j * phase)).astype(np.complex64), hop_length=256, length=len(y))
        return yd.astype(np.float32, copy=False)
    except Exception:
        return y


def _bytes_frames(y16: bytes, sr: int, frame_ms: int) -> List[bytes]:
    n = int(sr * frame_ms / 1000) * 2
    return [y16[i:i+n] for i in range(0, len(y16) - (len(y16) % n), n)]


def _vad_segments(y: np.ndarray, sr: int, aggr: int, frame_ms: int, pad_ms: int) -> List[Tuple[int,int]]:
    assert _HAS_VAD, "webrtcvad 未安装"
    y = y.astype(np.float32, copy=False)
    vad = webrtcvad.Vad(int(aggr))
    s16 = (y * 32767.0).clip(-32768, 32767).astype(np.int16).tobytes()
    frames = _bytes_frames(s16, sr, frame_ms)
    hop = int(sr * frame_ms / 1000)
    pad = int(sr * pad_ms / 1000)

    segs = []
    cur = None
    for i, f in enumerate(frames):
        if len(f) != len(frames[0]): break
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
    y = y.astype(np.float32, copy=False)
    n = min(int(sr*ms/1000), len(y)//2)
    if n<=0: return y
    fade_in = np.linspace(0,1,n, dtype=np.float32)
    fade_out= np.linspace(1,0,n, dtype=np.float32)
    y[:n] *= fade_in; y[-n:] *= fade_out
    return y


def _energy(x: np.ndarray) -> float:
    x = x.astype(np.float32, copy=False)
    return float(np.mean(x**2))


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
    y = y.astype(np.float32, copy=False)
    if not segs:
        return y[:int(want_sec*sr)]
    scored = []
    for s,e in segs:
        seg = y[s:e].astype(np.float32, copy=False)
        if len(seg) == 0: 
            continue
        if prefer_harmonic:
            score = 0.7*_hnr_score(seg, sr) + 0.3*_energy(seg)
        else:
            score = _energy(seg)
        scored.append((score, s, e))
    if not scored:
        return y[:int(want_sec*sr)]
    scored.sort(key=lambda t: t[0], reverse=True)

    need = int(want_sec*sr)
    got = 0
    out = []
    xf = int(sr*crossfade_ms/1000)
    for _, s, e in scored:
        if got >= need: break
        seg = y[s:e].astype(np.float32, copy=False)
        take = min(len(seg), need-got)
        seg = seg[:take]
        if not out:
            out.append(seg)
        else:
            a = out[-1]
            if xf>0 and len(a)>xf and len(seg)>xf:
                fo = np.linspace(1,0,xf, dtype=np.float32)
                fi = np.linspace(0,1,xf, dtype=np.float32)
                a[-xf:] = a[-xf:]*fo + seg[:xf]*fi
                out[-1] = a
                out.append(seg[xf:])
            else:
                out.append(seg)
        got += take
    return np.concatenate(out).astype(np.float32, copy=False) if out else y[:need]


def _snr_estimate(y: np.ndarray, sr: int, segs: List[Tuple[int,int]]) -> float:
    """基于 VAD：speech 段能量 / 非语音段能量；不足时用分位数回退"""
    y = y.astype(np.float32, copy=False)
    mask = np.zeros(len(y), dtype=bool)
    for s,e in segs:
        mask[s:e] = True
    speech = y[mask]
    noise = y[~mask]
    if len(speech)<sr*0.2 or len(noise)<sr*0.2:
        # fallback：分位数估计
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)).astype(np.float32, copy=False)
        sp = np.quantile(S, 0.80); ns = np.quantile(S, 0.20)+1e-8
        return float(20*np.log10(sp/ns))
    ps = float(np.mean(speech**2))+1e-12
    pn = float(np.mean(noise**2))+1e-12
    return float(10*np.log10(ps/pn))


def _auto_detect_hum(y: np.ndarray, sr: int) -> Optional[int]:
    """简易嗡声检测：50/60Hz 处幅度峰值更高则返回对应频率"""
    y = y.astype(np.float32, copy=False)
    try:
        S = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024)).mean(axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
        def band_power(f0, bw=3):
            idx = np.where((freqs>=f0-bw)&(freqs<=f0+bw))[0]
            return float(S[idx].mean()) if len(idx)>0 else 0.0
        p50, p60 = band_power(50), band_power(60)
        if max(p50, p60) < 0.001:
            return None
        return 50 if p50 >= p60 else 60
    except Exception:
        return None


def preprocess_reference_dual(
    in_path: str,
    out_dir: str,
    cfg: DualRefCfg = DualRefCfg()
) -> Tuple[str, str]:
    """
    读取参考音频 → 轻/重两条参考：
    - embed_ref.wav（轻处理，供音色/相似度抽取）
    - asr_ref.wav   （重处理，供ASR/对齐/可懂性）
    返回 (embed_ref_path, asr_ref_path)
    """
    os.makedirs(out_dir, exist_ok=True)
    y, sr = _load_audio(in_path, cfg.target_sr)

    # 限时
    if cfg.max_in_seconds and len(y) > cfg.max_in_seconds*sr:
        y = y[:int(cfg.max_in_seconds*sr)].astype(np.float32, copy=False)

    # 初步滤波（轻）
    y0 = _apply_filters(y, sr, cfg.highpass_hz, cfg.lowpass_hz).astype(np.float32, copy=False)

    # VAD
    if cfg.use_vad and _HAS_VAD:
        segs = _vad_segments(y0, sr, cfg.vad_aggr, cfg.vad_frame_ms, cfg.vad_pad_ms)
    else:
        segs = [(0, len(y0))]

    # SNR 评估（用于决策）
    snr = _snr_estimate(y0, sr, segs)

    # 自动 50/60Hz 嗡声陷波（若需要）
    hum = cfg.hum_notch_hz
    if hum is None:
        hum = _auto_detect_hum(y0, sr)
    if hum in (50, 60):
        y0 = _apply_notch(y0, sr, hum)

    # —— 声纹参考（轻处理）——
    y_embed = _pick_segments(y0, sr, segs, cfg.pick_seconds, cfg.prefer_harmonic, cfg.crossfade_ms)
    if snr < cfg.snr_mild_th:
        strength = cfg.denoise_strength_embed_light if snr < cfg.snr_bad_th else cfg.denoise_strength_embed_mild
        y_embed = _simple_denoise(y_embed, sr, strength=float(strength))
    y_embed = _fade_io(y_embed, sr, cfg.fade_ms)
    y_embed = _peak_norm(y_embed, cfg.peak_dbfs)
    if cfg.rms_dbfs_embed is not None:
        y_embed = _rms_norm(y_embed, cfg.rms_dbfs_embed)
    embed_path = os.path.join(out_dir, "embed_ref.wav")
    sf.write(embed_path, y_embed.astype(np.float32, copy=False), sr)

    # —— ASR 参考（重处理）——
    y_asr = _pick_segments(y0, sr, segs, cfg.pick_seconds, cfg.prefer_harmonic, cfg.crossfade_ms)
    y_asr = _simple_denoise(y_asr, sr, strength=float(cfg.denoise_strength_asr))
    y_asr = _apply_filters(y_asr, sr, cfg.highpass_hz or 60, min(cfg.lowpass_hz or 8000, 7500))
    y_asr = _fade_io(y_asr, sr, cfg.fade_ms)
    y_asr = _peak_norm(y_asr, cfg.peak_dbfs)
    if cfg.rms_dbfs_asr is not None:
        y_asr = _rms_norm(y_asr, cfg.rms_dbfs_asr)
    asr_path = os.path.join(out_dir, "asr_ref.wav")
    sf.write(asr_path, y_asr.astype(np.float32, copy=False), sr)

    return embed_path, asr_path
