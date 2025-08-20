# ref_preproc_dual.py
# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfilt, iirnotch, filtfilt, lfilter

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
    hum_notch_hz: Optional[int] = None # 50 或 60（自动检测亦可见下方）

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


def _ensure_mono(y: np.ndarray) -> np.ndarray:
    return y.mean(axis=0) if y.ndim == 2 else y


def _load_audio(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=False)
    y = (y.mean(axis=0) if y.ndim == 2 else y).astype(np.float32)
    #y = _ensure_mono(y.astype(np.float32))
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, fix=True)
        sr = target_sr
    return y, sr


def _peak_norm(y: np.ndarray, peak_dbfs: float) -> np.ndarray:
    peak = float(np.max(np.abs(y))) + 1e-9
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
    if hp:
        y = sosfilt(_butter_sos("high", hp, sr), y)
    if lp and lp < int(sr * 0.49):
        y = sosfilt(_butter_sos("low", lp, sr), y)
    return y.astype(np.float32)


# def _apply_notch(y: np.ndarray, sr: int, freq: int) -> np.ndarray:
#     # 陷波 50/60Hz（Q=30）
#     b, a = iirnotch(w0=freq/(sr/2), Q=30)
#     return librosa.lfilter(b, a, y).astype(np.float32)

def _apply_notch(y: np.ndarray, sr: int, freq: int) -> np.ndarray:
    """
    50/60 Hz 嗡声陷波。优先零相位 filtfilt；当音频太短时回退到 lfilter。
    """
    # 保护：目标频率必须小于奈奎斯特频率
    if freq >= sr / 2:
        return y.astype(np.float32)

    w0 = freq / (sr / 2.0)           # 归一化频率
    b, a = iirnotch(w0=w0, Q=30.0)   # Q 可以按需调大以更窄

    # filtfilt 需要足够点数；不够则用 lfilter 兜底避免报错
    pad_needed = 3 * max(len(a), len(b))
    if len(y) > pad_needed:
        y_f = filtfilt(b, a, y)
    else:
        y_f = lfilter(b, a, y)

    return y_f.astype(np.float32)





def _simple_denoise(y: np.ndarray, sr: int, strength: float) -> np.ndarray:
    if _HAS_NR:
        try:
            return nr.reduce_noise(y=y, sr=sr, stationary=False)
        except Exception:
            pass
    # 轻量谱减
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) + 1e-8
    noise = np.quantile(S, 0.10, axis=1, keepdims=True)
    Sd = np.maximum(S - strength * noise, 0.0)
    phase = np.angle(librosa.stft(y, n_fft=1024, hop_length=256))
    yd = librosa.istft(Sd * np.exp(1j * phase), hop_length=256, length=len(y))
    return yd.astype(np.float32)


def _bytes_frames(y16: bytes, sr: int, frame_ms: int) -> List[bytes]:
    n = int(sr * frame_ms / 1000) * 2
    return [y16[i:i+n] for i in range(0, len(y16), n)]


def _vad_segments(y: np.ndarray, sr: int, aggr: int, frame_ms: int, pad_ms: int) -> List[Tuple[int,int]]:
    assert _HAS_VAD, "webrtcvad 未安装"
    vad = webrtcvad.Vad(aggr)
    s16 = (y * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
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
    if cur is not None: segs.append(tuple(cur))

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
    if n<=0: return y
    fade_in = np.linspace(0,1,n, dtype=np.float32)
    fade_out= np.linspace(1,0,n, dtype=np.float32)
    y[:n] *= fade_in; y[-n:] *= fade_out
    return y


def _energy(x: np.ndarray) -> float:
    return float(np.mean(x**2))


# def _hnr_score(x: np.ndarray, sr: int) -> float:
#     # 粗略 HNR：Harmonic/RMS after HPSS
#     H, P = librosa.effects.hpss(librosa.stft(x, n_fft=1024, hop_length=256))
#     h = np.mean(np.abs(H)); p = np.mean(np.abs(P)) + 1e-8
#     return float(h/p)

# 2) HNR 打分 —— 确保输入是 float32 再喂给 librosa
def _hnr_score(x: np.ndarray, sr: int) -> float:
    x = x.astype(np.float32, copy=False)
    H, P = librosa.effects.hpss(librosa.stft(x, n_fft=1024, hop_length=256))
    h = np.mean(np.abs(H))
    p = np.mean(np.abs(P)) + 1e-8
    return float(h / p)


def _pick_segments(y: np.ndarray, sr: int, segs: List[Tuple[int,int]], want_sec: float, prefer_harmonic: bool, crossfade_ms: float) -> np.ndarray:
    if not segs:
        return y[:int(want_sec*sr)]
    scored = []
    for s,e in segs:
        seg = y[s:e]
        if prefer_harmonic:
            score = 0.7*_hnr_score(seg, sr) + 0.3*_energy(seg)
        else:
            score = _energy(seg)
        scored.append((score, s, e))
    scored.sort(key=lambda t: t[0], reverse=True)

    need = int(want_sec*sr)
    got = 0
    out = []
    xf = int(sr*crossfade_ms/1000)
    for _, s, e in scored:
        if got >= need: break
        seg = y[s:e]
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
    return np.concatenate(out) if out else y[:need]


def _snr_estimate(y: np.ndarray, sr: int, segs: List[Tuple[int,int]]) -> float:
    """基于 VAD：speech 段能量 / 非语音段能量"""
    mask = np.zeros(len(y), dtype=bool)
    for s,e in segs:
        mask[s:e] = True
    speech = y[mask]
    noise = y[~mask]
    if len(speech)<sr*0.2 or len(noise)<sr*0.2:
        # fallback：分位数估计
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
        sp = np.quantile(S, 0.80); ns = np.quantile(S, 0.20)+1e-8
        return 20*np.log10(sp/ns)
    ps = np.mean(speech**2)+1e-12
    pn = np.mean(noise**2)+1e-12
    return 10*np.log10(ps/pn)


def _auto_detect_hum(y: np.ndarray, sr: int) -> Optional[int]:
    # 简易嗡声检测：50/60Hz 处幅度峰值更高则返回对应频率
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024)).mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    def band_power(f0, bw=3):
        idx = np.where((freqs>=f0-bw)&(freqs<=f0+bw))[0]
        return float(S[idx].mean()) if len(idx)>0 else 0.0
    p50, p60 = band_power(50), band_power(60)
    if max(p50, p60) < 0.001:  # 很弱就不做
        return None
    return 50 if p50 >= p60 else 60


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
        y = y[:int(cfg.max_in_seconds*sr)]

    # 初步滤波（轻）
    y0 = _apply_filters(y, sr, cfg.highpass_hz, cfg.lowpass_hz)

    # VAD
    if cfg.use_vad and _HAS_VAD:
        segs = _vad_segments(y0, sr, cfg.vad_aggr, cfg.vad_frame_ms, cfg.vad_pad_ms)
    else:
        segs = [(0, len(y0))]

    # SNR 评估（用于决策）
    snr = _snr_estimate(y0, sr, segs)

    # 自动 50/60Hz 嗡声陷波（如果明显）
    hum = cfg.hum_notch_hz or _auto_detect_hum(y0, sr)
    if hum in (50, 60):
        y0 = _apply_notch(y0, sr, hum)

    # —— 声纹参考（轻处理）——
    y_embed = _pick_segments(y0, sr, segs, cfg.pick_seconds, cfg.prefer_harmonic, cfg.crossfade_ms)
    # 去噪决策：SNR 好就不去噪；边缘情况用很轻强度
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
    y_asr = _pick_segments(y0, sr, segs, cfg.pick_seconds, cfg.prefer_harmonic, cfg.crossfade_ms)
    y_asr = _simple_denoise(y_asr, sr, strength=cfg.denoise_strength_asr)
    # 更紧的带宽（有助于 ASR 可懂性）：可选再次高通/低通
    y_asr = _apply_filters(y_asr, sr, cfg.highpass_hz or 60, min(cfg.lowpass_hz or 8000, 7500))
    y_asr = _fade_io(y_asr, sr, cfg.fade_ms)
    y_asr = _peak_norm(y_asr, cfg.peak_dbfs)
    if cfg.rms_dbfs_asr is not None:
        y_asr = _rms_norm(y_asr, cfg.rms_dbfs_asr)
    asr_path = os.path.join(out_dir, "asr_ref.wav")
    sf.write(asr_path, y_asr, sr)

    return embed_path, asr_path

# “双路参考 + 动态决策”上：

# 相似度（40%）优先 → 给声纹/音色提取一条“轻处理 / 保真”参考（只做裁剪、VAD 选段、响度/高低通、极轻的去噪或直接不去噪）。

# 可懂性（40%）次优 → 给 ASR/标点/对齐一条“重处理 / 更干净”参考（更强的去噪/带通、RMS 归一），这条不参与音色克隆。

# 自然度（20%） → 两路都做端点淡入淡出、避免“金属音”的过度去噪；宁轻勿重。

# 下面给你一套可直接落地的工具模块（一个 py 文件），实现：

# 自动评估参考音频SNR与谐噪比(HNR)，动态决定“是否/多强去噪”。

# VAD 抽段（WebRTC），优先选“人声更稳定”的段，拼成 ~6 秒参考。

# 双路输出：embed_ref.wav（克隆用，轻处理） + asr_ref.wav（识别/对齐用，重处理）。

# 可选“50/60 Hz 嗡声”检测+陷波处理。

# 所有依赖均是常见包：webrtcvad librosa soundfile scipy numpy（可选 noisereduce）




# 依赖安装（任选其一环境里执行）：

# pip install webrtcvad librosa soundfile scipy numpy
# # 可选更稳的去噪
# pip install noisereduce
# 主脚本如何接入（以你的 run_0817.py 为例）
# 在读 CSV/准备每条任务前，对参考音频跑一次双路预处理，并把输出分别用于音色克隆与ASR/对齐：

# from ref_preproc_dual import preprocess_reference_dual, DualRefCfg

# # …读取一行任务 row 后：
# ref_in  = os.path.join("aigc_speech_generation_tasks", row["reference_speech"])
# ref_dir = os.path.join(".cache", "refproc", row["utt"])
# cfg = DualRefCfg(
#     target_sr=16000,
#     pick_seconds=6.0,
#     max_in_seconds=60.0,
#     vad_aggr=2,              # 噪声特别大可临时改 3
#     highpass_hz=60,
#     lowpass_hz=8000,
#     hum_notch_hz=None        # 自动检测 50/60Hz 嗡声
# )

# embed_ref, asr_ref = preprocess_reference_dual(ref_in, ref_dir, cfg)



# # 声纹/音色克隆 → 用 embed_ref
# # ASR/标点/辅助对齐 → 用 asr_ref（若你的流程里有ASR）
# # 例如：
# #   IndexTTS/F5/XTTS 的“参考音色音频” = embed_ref
# #   Whisper/Faster-Whisper/ASR 端 = asr_ref



# 你当前是“分段合成”，embed_ref 不要跟着每句变；在循环外或每个任务只做一次，然后全程复用。
# 如果流程中没有单独的 ASR 步（比如直接 TTS），也可以只用 embed_ref。

# 为什么这么做（和你贴的博客一致）
# 相似度（40%）：
# 只做轻处理（VAD 选段 + 统一采样/响度 + 轻滤波），尽量不去噪或极轻去噪。去噪会抹掉声纹的高频细节和谐波结构，损相似度。

# 可懂性（40%）：
# 给 ASR 一条重处理参考（强谱减、带通、RMS 归一），它不影响声纹，只提升识别/标点/断句稳定性。

# 自然度（20%）：
# 所有输出做端点淡入淡出，避免爆音/拼接噪声；不过度去噪，避免“金属音”。

# 动态决策：
# 我们用 VAD 区分语音/静音，估计 SNR；再用 HNR（谐波成分比）排序语音片段，挑“音质更稳定的人声”去拼接。

# SNR 高（≥15 dB）：embed_ref 基本不去噪；

# SNR 中（8–15 dB）：embed_ref 只很轻去噪；

# SNR 低（<8 dB）：embed_ref 仍然轻度（保护相似度），ASR 参考加重去噪。

# 嗡声：自动探测 50/60Hz，必要时加陷波，这类稳态噪最破坏可懂性但对音色影响小。

# 能否引入 ClearerVoice-Studio / MegaTTS3 文中做法？
# ClearerVoice-Studio（重去噪/分离）：建议只用于 ASR 参考或赛外的“预清洗”，不要喂给声纹抽取。极脏样本（风扇/街噪）可先离线跑一遍，把其输出再喂给我们这条管线生成 asr_ref.wav。

# MegaTTS3 文中的滤波/响度：我们已包含高/低通 + RMS -23 LUFS 近似（RMS dBFS）；如果你要更“广播级 LUFS”，可把 pyloudnorm 接入，把 RMS 改成 LUFS 归一，但比赛里意义不大。

# 分段/清洗/批处理：我们内部有 VAD+选段，避免把键盘声、环境声学成“风格”。若样本极端复杂，可考虑加一层说话人提取/分离（CV-Studio 的 target speaker extraction）→ 再进本模块。

# 小贴士（落地调优）
# 时间：参考总时长 6–8 秒通常最稳；更长并不一定更好。

# VAD 激进度：2 为折中，脏样本可临时 3；但 3 可能“切字”。

# 缓存：把 embed_ref.wav、asr_ref.wav 缓存到 .cache/refproc/<utt>/，重复跑时直接命中。

# 评估：若你有 ECAPA/ResNet 说话人模型，可对“原参考 vs 轻处理参考”的余弦相似度做个 sanity check，确保轻处理不伤声纹。

