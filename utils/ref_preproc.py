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


def _load_audio(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=False)
    y = _ensure_mono(y.astype(np.float32))
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, fix=True)
        sr = target_sr
    return y, sr


def _peak_normalize(y: np.ndarray, peak_dbfs: float) -> np.ndarray:
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

# 特点：

# 统一加载（单声道、目标采样率）、轻量降噪（可选）、高/低通滤波、RMS响度归一化

# WebRTC VAD 语音段提取（可选，默认开），自动挑“最干净的若干段”拼成 6–8 秒参考

# 时长上限裁剪、端点淡入淡出，避免爆音/不连续噪声

# 纯 Python + webrtcvad / librosa / scipy / soundfile / numpy 即可跑（可选 noisereduce）

# 需要的 pip：pip install webrtcvad librosa soundfile scipy numpy（可选）noisereduce

# 主脚本如何接入
# 你的 run_0817.py 里，当前已经有「重采样 + 响度归一化 + 最长 60 秒裁剪」的思路，
# 并通过 cut_reference(..., max_sec=60) 做了参考时长限制。 另外，循环里会先 trim/cut 再送合成。

# 把这些合到新工具即可。示例最小改动：

# python

# # run_0817.py 中（开头）
# from ref_preproc import preprocess_reference, RefPreprocCfg

# # …解析你的命令行参数后，构造配置（示例）
# cfg = RefPreprocCfg(
#     target_sr=16000,
#     do_denoise=True,        # 轻量降噪（避免把环境噪声当作说话人风格）
#     vad_aggr=2,             # 2 较稳；若噪声多可试 3
#     pick_seconds=6.0,       # 最终拼成 ~6秒参考
#     max_seconds=60.0,       # 参考 wav 最长只看 1 分钟
# )

# # 在进入合成循环前/或每题目里，对当前参考语音做一次预处理并缓存
# ref_in = f"./aigc_speech_generation_tasks/{row.reference_speech}"
# ref_out = f".cache/refproc/{row.utt}_ref16k.wav"
# _, sr, saved = preprocess_reference(ref_in, out_path=ref_out, cfg=cfg)

# # 之后把 ref_out 作为你 IndexTTS/F5/XTTS 的参考音频路径传入
# # e.g. f5-tts_infer-cli --ref_audio "{ref_out}" ...
# 如果你有“分句合成”的流程，参考音频只需要处理一次（全局复用）；不要在每个分句上重复做降噪/VAD（省时且避免引入不一致）。

# 3) 我们到底做了哪些处理（为什么对比赛有帮助）
# WebRTC VAD 选段：把纯语音（无静音/乐声/键盘声）挑出来，再从中选“能量更高/更稳定”的段拼成 6–8 秒“说话人特征摘要”。
# 这能显著降低“背景噪声被模型学成风格”的风险。WebRTC VAD 是工业级轻量算法，很多实时系统在用。
# Medium
# docs.cognitive-ml.fr

# 轻量降噪：默认采用谱减/noisereduce 的自适应方法；无需大模型，也能去掉稳态嗡嗡声、风扇声、嘶声。若你愿意引入更强的
# 实时降噪（RNNoise、DeepFilterNet、Facebook Denoiser/Demucs），可以在进入本模块之前做一遍“重降噪”作为预处理管线。

# 高/低通滤波：高通 50 Hz 去电源/风噪，低通 8 kHz 去高频嘶声，兼顾保真与稳定性（目标主要是声纹/韵律，不是 Hi-Fi）。

# 响度规范化：峰值限制在 -3 dBFS，RMS 拉到 -23 dBFS 左右，保证说话人嵌入/风格提取稳定，不会因为“录得太小/太大”而失真。

# 端点淡入淡出：避免爆音/咔哒声。这在你“分句拼接”流程里尤其重要（不连续噪声最容易被评分模型抓出来）。

# 可选的“进阶增强位点”
# 这些在比赛中经常能多拿几分（按需挑）：

# 更强的降噪前置（赛外、离线跑）：

# RNNoise（CPU 即可，低开销，实时），DeepFilterNet（低复杂度全带宽），Facebook Denoiser（Demucs 架构，质量高）。
# 在预处理前先跑一次，这样我们再做 VAD/滤波就更干净了。

# 设备/场景一致性裁剪：有的参考是“户外/键盘声/风扇恒噪”，尽量挑稳定人声的片段；我的 _pick_top_segments 已按能量优先，你也可以改成“能量/过零率/谱熵”组合评分。

# 限制参考总时长：6–8 秒对大多数零样本声纹模型很稳；过短不稳、过长会把噪声纳进来；你已经在原脚本里做了 60 秒上限裁剪（这里保留），并在 VAD 后再聚焦到 6 秒。

# 频带一致性：如果你的 TTS 模型在 16 kHz 训练/推理，那参考始终转到 16 kHz；不要混 44.1/48 kHz（声码器/前端会不稳）。

# 参考资料（可进一步加固）


# 你提供的 Notion/Juejin 链接我这边直接打开受限，不过从经验和上面公开资料总结的建议已覆盖“VAD 抽段 + 降噪 + 
# 统一采样率 + 规范化 + 淡入淡出”的关键点。如果你愿意贴出其中关键段落，我可以继续把要点合进来。

# 小结 & 落地步骤
# 把上面的 ref_preproc.py 放进项目（如 utils/ref_preproc.py）。

# pip install webrtcvad librosa soundfile scipy numpy（可选 noisereduce）。

# 在主脚本引入并调用 preprocess_reference(...)，把返回的 ref_out 路径交给 IndexTTS / F5 / Piper。

# 想再卷一点：离线先用 RNNoise / DeepFilterNet / Denoiser 做“重降噪”，再跑本文管线。这样能显著降低
# “噪声风格化”的风险，分数更稳。