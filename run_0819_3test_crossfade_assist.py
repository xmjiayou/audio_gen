#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 例：当前已激活 tts22
# python run_0817_3.py \
#   --team_name MyTeam \
#   --k_idx 2 --k_xtts 1 --k_f5 1 \
#   --use_xtts_if_cached \
#   --use_piper_if_available --piper_model models/piper/zh_CN-huayan-medium.onnx \
#   --asr_backend faster \
#   --python_aasist /opt/venvs/aasist/bin/python \
#   --aasist_repo /path/to/SSL_Anti-spoofing \
#   --aasist_ckpt /path/to/best_SSL_model_LA.pth

"""
ALWAYS-200 SAFE RUNNER (融合版)
- 主力: IndexTTS (k_idx 多候选)
- 备选: [可选且仅当本地缓存存在] XTTS v2 (k_xtts)
- 兜底: F5-TTS (k_f5)
- 离线兜底(可选): Piper (纯离线中文 TTS, 需自行准备 onnx)
- 最终兜底: 参考拼贴(ref collage) → 哔声占位(tone)
- 工程: 分句合成 + 10ms 交叉淡化 + 端点淡入淡出 + 轻 dither
- 评分: WER(ASR) + ECAPA 说话人相似度 + [可选]AASIST 反欺诈
- 稳定性: 生成过程不中断，最后扫尾保证 EXACTLY 200 个 WAV + CSV + ZIP
"""

import os, sys, shlex, shutil, subprocess, zipfile, warnings, gc, tempfile, random
from pathlib import Path
import argparse
import numpy as np
from utils.num_cn_normalizer import normalize_text
from utils.ref_audio_trim import cut_reference

from utils.resume_utils import list_official_wavs, summarize_progress ##########################
from utils.ref_preproc3 import preprocess_reference, RefPreprocCfg

import json, subprocess

# === NEW: 边界等功率交叉淡化 & 统一后处理所需依赖 ===

import scipy.signal as ss

import soundfile as sf  # 若本文件里已 import，则无需重复

import pandas as pd, soundfile as sf, librosa
import pyloudnorm as pyln

from tqdm import tqdm
from jiwer import wer
import torch
#from speechbrain.pretrained import EncoderClassifier
try:
    from speechbrain.inference import EncoderClassifier  # SB ≥ 1.0
except Exception:
    from speechbrain.pretrained import EncoderClassifier  # 兼容老版本

def _call_aasist_external(py_exe: str, cli_py: str, repo: str, ckpt: str, wav_path: str) -> float:
    """
    调用另一个虚拟环境里的 AASIST CLI，stdout 返回 JSON：{"cm_score": 0.123}
    """
    cmd = [py_exe, cli_py, "--repo", repo, "--ckpt", ckpt, "--wav", wav_path]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    # CLI 会把日志打到 stderr 或 stdout，这里只要能解析到最后一行 JSON 即可
    last_line = out.strip().splitlines()[-1]
    try:
        obj = json.loads(last_line)
        return float(obj["cm_score"])
    except Exception:
        # 兜底：从整体文本里搜一遍第一个 JSON
        import re
        m = re.search(r'\{.*"cm_score"\s*:\s*([0-9\.eE+-]+).*?\}', out)
        if m: 
            return float(m.group(1))
        raise RuntimeError(f"External AASIST failed, output=\n{out}")

class ExternalAASISTScorer:
    """提供 .score(path) 接口，便于与 pick_best 现有逻辑对接"""
    def __init__(self, py_exe: str, cli_py: str, repo: str, ckpt: str):
        self.py = py_exe
        self.cli = cli_py
        self.repo = repo
        self.ckpt = ckpt
    def score(self, wav_path):
        return _call_aasist_external(self.py, self.cli, self.repo, self.ckpt, str(wav_path))

# ------------------------ small helper: lazy installer ------------------------
def ensure_base():
    #pkgs = ["pandas","tqdm","soundfile","librosa","speechbrain","jiwer"]
    pkgs = ["pandas","tqdm","soundfile","librosa","speechbrain","jiwer","pyloudnorm"]

    import importlib, subprocess as sp
    miss=[]
    for p in pkgs:
        try: importlib.import_module(p)
        except Exception: miss.append(p)
    if miss:
        print("Installing base deps:", miss)
        sp.check_call([sys.executable, "-m", "pip", "install", *miss])
ensure_base()




# ------------------------ env / paths ------------------------
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(Path("./.hf_cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("./.cache").resolve()))

#CSV_PATH = "aigc_speech_generation_tasks/aigc_speech_generation_tasks.csv" ######
#REF_DIR  = Path("aigc_speech_generation_tasks")
REF_DIR  = Path("tests2")
PRE_DIR  = Path("preproc_refs"); PRE_DIR.mkdir(exist_ok=True)

# ------------------------ audio utils ------------------------
def loudness_norm(y, target_db=-23.0):
    rms = np.sqrt((y**2).mean() + 1e-9); ref = 10 ** (target_db/20)
    if rms < 1e-6: return y
    return np.clip(y * (ref/rms), -1.0, 1.0)

def preprocess_ref(wav_path: Path, target_sr=16000) -> Path:
    out = PRE_DIR / wav_path.name
    if out.exists(): return out
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    if sr != target_sr: y = librosa.resample(y, orig_sr=sr, target_sr=target_sr); sr=target_sr
    y = loudness_norm(y, -23.0)
    sf.write(out, y, sr)
    return out


def seg_loudness_norm(y: np.ndarray, target_db=-23.0):
    rms = float(np.sqrt(np.mean(np.square(y)) + 1e-9))
    ref = 10 ** (target_db / 20.0)
    if rms < 1e-6:
        return y
    g = ref / rms
    return np.clip(y * g, -1.0, 1.0).astype(np.float32)

def lufs_normalize(y: np.ndarray, sr: int, target_lufs: float = -23.0, meter: pyln.Meter | None = None) -> np.ndarray:
    """
    使用 ITU-R BS.1770 / LUFS 做响度归一。meter 如果传入会被复用（避免多次构建）。
    """
    y = y.astype(np.float32)
    m = meter or pyln.Meter(sr)
    loud = m.integrated_loudness(y)
    # 防止静音 / 非数
    if not np.isfinite(loud):
        return y
    return pyln.normalize.loudness(y, loud, target_lufs).astype(np.float32)

# === NEW: 用上一段尾巴与当前段开头做“包络相关”对齐，裁掉回看音频 ===


def _mel_envelope(y: np.ndarray, sr: int, n_fft=512, hop=160, n_mels=40) -> tuple[np.ndarray, int]:
    """log-mel 能量随时间的包络（标准化），以及 hop 长度（样点）"""
    S = librosa.feature.melspectrogram(y=y.astype(np.float32), sr=sr, n_fft=n_fft,
                                       hop_length=hop, n_mels=n_mels, power=1.0)
    env = np.log(S + 1e-6).mean(axis=0).astype(np.float32)
    env = (env - env.mean()) / (env.std() + 1e-8)
    return env, hop

def _align_cut_by_prev_tail(prev_tail: np.ndarray, y: np.ndarray, sr: int,
                            max_search_sec: float = 2.0, min_sim: float = 0.35) -> int | None:
    """
    用上一段尾巴 prev_tail 与当前段 y 的开头做对齐。
    返回应当裁掉的样点数（cut_samples）；若对齐不可靠，返回 None.
    """
    if prev_tail is None: 
        return None
    if len(prev_tail) < int(0.15 * sr) or len(y) < int(0.20 * sr):
        return None

    # 只在当前段开头一定范围内搜索
    head = y[: int(max_search_sec * sr)].astype(np.float32)

    env_prev, hop = _mel_envelope(prev_tail, sr)
    env_head, _ = _mel_envelope(head, sr)

    # 长度不足以做匹配
    if len(env_head) <= len(env_prev) + 2:
        return None

    # 相关匹配（等价于在 env_head 上滑动 env_prev）
    corr = ss.correlate(env_head, env_prev, mode="valid")
    idx = int(np.argmax(corr))

    # 归一化相似度检查（防止假高峰）
    win = env_head[idx: idx + len(env_prev)]
    if len(win) != len(env_prev):
        return None
    sim = float(np.dot(win, env_prev) /
                (np.linalg.norm(win) * np.linalg.norm(env_prev) + 1e-8))
    if sim < min_sim:
        return None

    # 把帧位移换算为样点；剪到“匹配点 + 上一段尾巴长度”略减一点点以避免重叠
    offset = idx * hop
    cut = offset + len(prev_tail) - int(0.02 * sr)  # 减去 ~20ms 的冗余
    cut = max(0, min(cut, len(y) - 1))
    return cut

def _choose_tail_window_seconds(prev_tail_text: str | None, sr: int) -> float:
    """
    根据 look-back 字符数动态决定跟随匹配用的尾巴时长（下限 0.4s，上限 1.0s）
    """
    if not prev_tail_text:
        return 0.6
    # 粗略：每个字 ~0.10s + 0.20s 基线
    sec = 0.20 + 0.10 * len(prev_tail_text)
    return float(max(0.4, min(1.0, sec)))

# —— 合成后按“look-back 文字长度”估计裁掉前导 —— 
def trim_head_by_chars(y: np.ndarray, sr: int, char_cnt: int, avg_char_sec: float = 0.12, bias_sec: float = 0.06):
    """根据回看的字符数估计需裁掉的前导秒数（避免把回看内容留在最终音频里）"""
    cut = int(sr * max(0.0, char_cnt * avg_char_sec + bias_sec))
    if cut <= 0 or cut >= len(y):
        return y
    return y[cut:].astype(np.float32)

def apply_fade(y, sr, fade_ms=10):
    n = max(1, int(sr*fade_ms/1000))
    if len(y) < 2*n: return y
    y[:n]  *= np.linspace(0,1,n,endpoint=True)
    y[-n:] *= np.linspace(1,0,n,endpoint=True)
    return y

# def concat_crossfade(chunks, sr, xfade_ms=10):
#     if not chunks: return np.zeros(0, dtype=np.float32)
#     xfade = int(sr*xfade_ms/1000)
#     out = chunks[0].copy()
#     for i in range(1,len(chunks)):
#         a, b = out, chunks[i]
#         if xfade>0 and len(a)>xfade and len(b)>xfade:
#             a_tail = a[-xfade:]; b_head = b[:xfade]
#             mix = a_tail*np.linspace(1,0,xfade) + b_head*np.linspace(0,1,xfade)
#             out = np.concatenate([a[:-xfade], mix, b[xfade:]], axis=0)
#         else:
#             out = np.concatenate([a, b], axis=0)
#     return np.clip(out, -1.0, 1.0)

# === NEW: 等功率交叉淡化 ===
def crossfade(prev: np.ndarray, curr: np.ndarray, sr: int, ms: int = 60) -> np.ndarray:
    """等功率交叉淡化（equal-power, 40–80ms 建议）"""
    n = int(sr * ms / 1000)
    n = min(n, len(prev), len(curr))
    if n <= 0:
        return np.concatenate([prev, curr], axis=0)
    # 等功率窗：cos/sin
    t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
    w_out = np.cos(t)
    w_in  = np.sin(t)
    x = prev.copy()
    x[-n:] = x[-n:] * w_out + curr[:n] * w_in
    return np.concatenate([x, curr[n:]], axis=0)


def concat_crossfade(chunks: list[np.ndarray], sr: int, xfade_ms: int = 60) -> np.ndarray:
    """改为调用等功率 crossfade；默认 60ms（可在调用处改 40–80ms）"""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    out = chunks[0].astype(np.float32)
    for i in range(1, len(chunks)):
        out = crossfade(out, chunks[i].astype(np.float32), sr, ms=xfade_ms)
    # 防止叠加后轻微溢出
    out = np.clip(out, -1.0, 1.0).astype(np.float32)
    return out
# === NEW: 统一后处理（轻高通 + 可选全局卷积混响） ===
def _highpass(y: np.ndarray, sr: int, cutoff_hz: float = 50.0, order: int = 2) -> np.ndarray:
    """极轻高通，去 DC / 低频隆隆，不要太激进"""
    # 使用 SOS 滤波器，稳定性更好
    sos = ss.butter(order, cutoff_hz / (sr * 0.5), btype="highpass", output="sos")
    y_hp = ss.sosfiltfilt(sos, y).astype(np.float32)
    return y_hp


def _convolve_reverb(y: np.ndarray, sr: int, ir_path: str | None = None, wet_db: float = -18.0) -> np.ndarray:
    """可选：全局同一 IR 作极轻混响；ir_path=None 则直接返回原始"""
    if not ir_path:
        return y
    try:
        ir, ir_sr = sf.read(ir_path, always_2d=False)
        if ir.ndim > 1:
            ir = ir.mean(axis=1)
        if ir_sr != sr:
            ir = librosa.resample(ir.astype(np.float32), orig_sr=ir_sr, target_sr=sr)
        # 归一化 IR 并设置湿度
        ir = ir.astype(np.float32)
        ir /= (np.max(np.abs(ir)) + 1e-9)
        wet = 10 ** (wet_db / 20.0)  # 负 dB，极轻
        tail = ss.fftconvolve(y, ir, mode="full")[: len(y)].astype(np.float32)
        out = y + wet * tail
        # 安全限幅（不改变响度标定）
        peak = np.max(np.abs(out)) + 1e-9
        if peak > 1.0:
            out = out / peak
        return out.astype(np.float32)
    except Exception:
        # 任何问题直接返回原始，不影响主流程
        return y


def split_sentences(text, max_len=80):
    PUNC_HARD = "。！？!?"
    PUNC_SOFT = "，,、;；:："
    sents, buf = [], ""
    for ch in str(text):
        buf += ch
        if ch in PUNC_HARD:
            sents.append(buf.strip()); buf=""
        elif len(buf)>=max_len and ch in PUNC_SOFT:
            sents.append(buf.strip()); buf=""
    if buf.strip(): sents.append(buf.strip())
    return [s for s in sents if s]

def estimate_duration(text: str):
    t = str(text)
    base = 0.18 * sum(1 for c in t if c.strip())
    bonus = 0.2 * sum(1 for c in t if c in "。！？!?")
    return max(0.6, min(15.0, base + bonus))

# ------------------------ ASR (faster/openai + device choice) ----------------
_faster_model = None
_openai_whisper = None
# --- NEW: simple character-level CER for Chinese ---
def _cer_char_level(ref: str, hyp: str) -> float:
    # remove spaces to be robust
    ref = "".join(ref.split())
    hyp = "".join(hyp.split())
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    # DP Levenshtein
    import numpy as np
    dp = np.zeros((len(ref)+1, len(hyp)+1), dtype=np.int32)
    for i in range(len(ref)+1): dp[i,0] = i
    for j in range(len(hyp)+1): dp[0,j] = j
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i,j] = min(dp[i-1,j]+1, dp[i,j-1]+1, dp[i-1,j-1]+cost)
    return float(dp[len(ref), len(hyp)]) / float(len(ref))

def asr_wer(wav_path: Path, ref_text: str, args, device=None):
    if args.no_asr:
        return 0.5, ""  # neutral
    chosen = getattr(args, "asr_device", "auto")
    dev = device or (chosen if chosen != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.asr_backend == "faster":
        try:
            import importlib, subprocess as sp
            try:
                import faster_whisper  # noqa
            except Exception:
                print("Installing faster-whisper ...")
                sp.check_call([sys.executable, "-m", "pip", "install", "faster-whisper", "huggingface_hub"])
            from faster_whisper import WhisperModel
            global _faster_model
            if _faster_model is None:
                _faster_model = WhisperModel(
                    args.asr_model, device=dev,
                    compute_type=args.asr_compute_type,
                    download_root=args.asr_download_root
                )
            segs, info = _faster_model.transcribe(str(wav_path), language="zh", beam_size=5)
            hyp = "".join([s.text for s in segs]).strip()
            if not hyp: return 1.0, ""
            #return wer(ref_text.strip(), hyp), hyp
            return _cer_char_level(ref_text.strip(), hyp), hyp
        except Exception as e:
            print("[ASR] faster-whisper failed, fallback to openai-whisper:", e)

    try:
        global _openai_whisper
        if _openai_whisper is None:
            import importlib, subprocess as sp
            try:
                import whisper  # noqa
            except Exception:
                print("Installing openai-whisper ...")
                sp.check_call([sys.executable, "-m", "pip", "install", "openai-whisper"])#whisperopenai-whisper
            import whisper as oaiw
            _openai_whisper = oaiw.load_model("large-v3", device=dev)
        result = _openai_whisper.transcribe(str(wav_path), language="zh")
        hyp = result.get("text","").strip()
        if not hyp: return 1.0, ""
        #return wer(ref_text.strip(), hyp), hyp
        return _cer_char_level(ref_text.strip(), hyp), hyp
    except Exception as e:
        print("[ASR] openai-whisper failed:", e)
        return 0.5, ""  # neutral
#     # inside asr_wer(...), in the openai-whisper block
#     try:
#         import whisper as oaiw
#     except Exception:
#         print("Installing openai-whisper ...")
#         sp.check_call([sys.executable, "-m", "pip", "install", "-U", "openai-whisper"])
#         import whisper as oaiw

#     def _load_openai(model_name, device):
#         try:
#             return oaiw.load_model(model_name, device=device)
#         except RuntimeError as e:
#             if "CUDA out of memory" in str(e):
#                 print("[ASR] OOM on", model_name, "→ falling back to medium on CPU")
#                 return oaiw.load_model("medium", device="cpu")
#             raise

#     _openai_whisper = _load_openai(args.asr_model or "large-v3", dev)


# ------------------------ Speaker similarity (ECAPA) --------------------------
_spk_m = None
def spk_similarity(ref_wav: Path, syn_wav: Path, device=None):
    global _spk_m
    if _spk_m is None:
        _spk_m = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device or ("cuda" if torch.cuda.is_available() else "cpu")}
        )
    def _emb(p):
        sig, sr = sf.read(p)
        if sig.ndim>1: sig = sig.mean(axis=1)
        if sr != 16000: sig = librosa.resample(sig, orig_sr=sr, target_sr=16000)
        sig = torch.from_numpy(sig).float().unsqueeze(0)
        with torch.no_grad():
            e = _spk_m.encode_batch(sig).squeeze(0).squeeze(0).cpu().numpy()
        return e / (np.linalg.norm(e)+1e-9)
    a, b = _emb(ref_wav), _emb(syn_wav)
    return float(np.dot(a,b))

# ------------------------ Optional AASIST ------------------------------------

import json, subprocess

def _call_aasist_external(py_exe: str, cli_py: str, repo: str, ckpt: str, wav_path: str) -> float:
    """
    调用另一个虚拟环境里的 AASIST CLI，stdout 返回 JSON：{"cm_score": 0.123}
    """
    cmd = [py_exe, cli_py, "--repo", repo, "--ckpt", ckpt, "--wav", wav_path]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    # CLI 会把日志打到 stderr 或 stdout，这里只要能解析到最后一行 JSON 即可
    last_line = out.strip().splitlines()[-1]
    try:
        obj = json.loads(last_line)
        return float(obj["cm_score"])
    except Exception:
        # 兜底：从整体文本里搜一遍第一个 JSON
        import re
        m = re.search(r'\{.*"cm_score"\s*:\s*([0-9\.eE+-]+).*?\}', out)
        if m: 
            return float(m.group(1))
        raise RuntimeError(f"External AASIST failed, output=\n{out}")



class ExternalAASISTScorer:
    """提供 .score(path) 接口，便于与 pick_best 现有逻辑对接"""
    def __init__(self, py_exe: str, cli_py: str, repo: str, ckpt: str):
        self.py = py_exe
        self.cli = cli_py
        self.repo = repo
        self.ckpt = ckpt
    def score(self, wav_path):
        return _call_aasist_external(self.py, self.cli, self.repo, self.ckpt, str(wav_path))

class AASISTSpoofScorer:
    def __init__(self, repo_dir: str, ckpt_path: str, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.repo = Path(repo_dir).resolve()
        self.ckpt = Path(ckpt_path).resolve()
        if not self.repo.exists() or not self.ckpt.exists():
            raise FileNotFoundError("AASIST repo/ckpt not found")
        sys.path.insert(0, str(self.repo))
        try:
            import model as ssl_model
            self.net = ssl_model.Model(self.device)
            state = torch.load(str(self.ckpt), map_location=self.device)
            self.net.load_state_dict(state, strict=False); self.net.eval()
        except Exception as e:
            raise RuntimeError(f"AASIST init failed: {e}")

    @torch.inference_mode()
    def score(self, wav_path: Path) -> float:
        import torch.nn.functional as F
        wav, sr = sf.read(wav_path)
        if wav.ndim>1: wav = wav.mean(axis=1)
        if sr != 16000: wav = librosa.resample(wav, sr, 16000)
        x = torch.from_numpy(wav).float().to(self.device)[None,None,:]
        out = self.net(x)
        if isinstance(out, (list,tuple)): out = out[-1]
        if out.ndim==1: out = out.unsqueeze(0)
        if out.shape[-1]==2:
            prob = F.softmax(out, dim=-1)[0,-1].item()
        else:
            prob = torch.sigmoid(out[0,0]).item()
        return float(prob)

# # ------------------------ Backends -------------------------------------------
# class IndexTTSBackend:
#     def __init__(self, model_dir="checkpoints", cfg_path="checkpoints/config.yaml", device=None):
#         from indextts.infer import IndexTTS
#         self.tts = IndexTTS(model_dir=model_dir, cfg_path=cfg_path, device=device or ("cuda" if torch.cuda.is_available() else "cpu"))

#     def synth_sentencewise(self, ref_wav: Path, text: str, out_wav: Path, seed: int=0):
#         torch.manual_seed(seed)
#         sents = split_sentences(text)
#         pieces=[]; sr=24000
#         for s in sents:
#             with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
#                 tmp_path = Path(tmp.name)
#             self.tts.infer(str(ref_wav), s, str(tmp_path))
#             y, sr1 = sf.read(tmp_path); os.remove(tmp_path)
#             if y.ndim>1: y=y.mean(axis=1)
#             pieces.append(apply_fade(y.astype(np.float32), sr1)); sr=sr1
#         out = concat_crossfade(pieces, sr, xfade_ms=10)
#         sf.write(out_wav, out, sr)
#         return out_wav

class IndexTTSBackend:
    def __init__(self, model_dir="checkpoints", cfg_path="checkpoints/config.yaml", device=None):
        from indextts.infer import IndexTTS
        self.tts = IndexTTS(model_dir=model_dir, cfg_path=cfg_path, device=device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def synth_sentencewise(self, ref_wav: Path, text: str, out_wav: Path, seed: int=0):
        torch.manual_seed(seed)
        sents = split_sentences(text)
        pieces=[]; sr=24000
        prev_tail_audio = None  # 新：保存上一段的尾巴音频用于对齐


        # 全局 LUFS 目标与 meter（按首段采样率初始化一次）
        target_lufs = -23.0
        lufs_meter = None
        # 文本 look-back 设置
        lookback_chars = 8
        prev_tail = ""

        for i, s in enumerate(sents):
            # 将上一句尾部少量字符作为上下文预热
            context = prev_tail
            in_text = (context + s) if context else s

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            # 正常合成
            self.tts.infer(str(ref_wav), in_text, str(tmp_path))

            y, sr1 = sf.read(tmp_path); os.remove(tmp_path)
            if y.ndim>1: y=y.mean(axis=1)
            y = y.astype(np.float32)

            # # 剪掉“回看”对应的前导
            # if context:
            #     y = trim_head_by_chars(y, sr1, char_cnt=len(context))
            # ---- 新：优先用“与上一段尾巴的对齐”来裁掉回看音频；不可靠时回退到字符估计 ----
            cut = None
            if 'prev_tail_audio' in locals() and prev_tail_audio is not None:
                cut = _align_cut_by_prev_tail(prev_tail_audio, y, sr1, max_search_sec=2.0, min_sim=0.35)
            if cut is not None and 0 < cut < len(y):
                y = y[cut:].astype(np.float32)
            elif context:
                y = trim_head_by_chars(y, sr1, char_cnt=len(context))

            # 段内响度归一 + 端点淡入淡出
            # y = seg_loudness_norm(y, -23.0)
            # y = apply_fade(y, sr1)
            # 段后统一 LUFS（全局 meter 仅初始化一次）
            
            if lufs_meter is None:
                lufs_meter = pyln.Meter(sr1)
            y = lufs_normalize(y, sr1, target_lufs, lufs_meter)

            # --- 新：缓存“上一段尾巴音频”用来做下一段对齐 ---
            tail_sec = _choose_tail_window_seconds(prev_tail, sr1)
            tail_len = min(len(y), int(tail_sec * sr1))
            prev_tail_audio = y[-tail_len:].copy()

            # 再做端点淡入淡出
            y = apply_fade(y, sr1)



            pieces.append(y); sr=sr1
            # 更新下次的回看尾巴
            prev_tail = s[-lookback_chars:] if len(s) > lookback_chars else s

        out = concat_crossfade(pieces, sr, xfade_ms=60)
        # === NEW: 统一后处理 ===
        out = _highpass(out, sr, cutoff_hz=50.0)             # 40–60 Hz 轻高通
        # 若需要极轻空间一致性，可指定统一的 IR 文件路径；不需要就保持 None
        REVERB_IR = None  # 例如 "irs/room_small.wav"
        out = _convolve_reverb(out, sr, ir_path=REVERB_IR, wet_db=-18.0)
        sf.write(out_wav, out, sr)
        return out_wav


# class F5TTSCLIBackend:
#     def __init__(self, model="F5TTS_v1_Base"):
#         self.model = model
#         if shutil.which("f5-tts_infer-cli") is None:
#             subprocess.check_call([sys.executable,"-m","pip","install","f5-tts"])

#     def synth_sentencewise(self, ref_wav: Path, text: str, out_wav: Path):
#         sents = split_sentences(text)
#         tmp_wavs=[]
#         for i, s in enumerate(sents):
#             cmd = (
#                 f'f5-tts_infer-cli --model {self.model} '
#                 f'--ref_audio {shlex.quote(str(ref_wav))} --gen_text {shlex.quote(s)}'
#             )
#             subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#             default_out = Path("testsf5/infer_cli_basic.wav")
#             if not default_out.exists():
#                 raise FileNotFoundError("F5-TTS output not found")
#             p = out_wav.parent / f"_tmp_f5_{out_wav.stem}_{i}.wav"
#             shutil.move(str(default_out), str(p))
#             tmp_wavs.append(p)
#         pieces=[]; sr=24000
#         for p in tmp_wavs:
#             y, sr1 = sf.read(p); p.unlink(missing_ok=True)
#             if y.ndim>1: y=y.mean(axis=1)
#             pieces.append(apply_fade(y.astype(np.float32), sr1)); sr=sr1
#         out = concat_crossfade(pieces, sr, xfade_ms=10)
#         sf.write(out_wav, out, sr)
#         return out_wav
class F5TTSCLIBackend:
    def __init__(self, model="F5TTS_v1_Base"):
        self.model = model
        if shutil.which("f5-tts_infer-cli") is None:
            subprocess.check_call([sys.executable,"-m","pip","install","f5-tts"])

    def synth_sentencewise(self, ref_wav: Path, text: str, out_wav: Path):
        sents = split_sentences(text)
        tmp_wavs=[]
        for i, s in enumerate(sents):
            cmd = (
                f'f5-tts_infer-cli --model {self.model} '
                f'--ref_audio {shlex.quote(str(ref_wav))} --gen_text {shlex.quote(s)}'
            )
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            default_out = Path("testsf5/infer_cli_basic.wav")
            if not default_out.exists():
                raise FileNotFoundError("F5-TTS output not found")
            p = out_wav.parent / f"_tmp_f5_{out_wav.stem}_{i}.wav"
            shutil.move(str(default_out), str(p))
            tmp_wavs.append(p)
        pieces=[]; sr=24000
        for p in tmp_wavs:
            y, sr1 = sf.read(p); p.unlink(missing_ok=True)
            if y.ndim>1: y=y.mean(axis=1)
            pieces.append(apply_fade(y.astype(np.float32), sr1)); sr=sr1
        out = concat_crossfade(pieces, sr, xfade_ms=60)
        out = _highpass(out, sr, cutoff_hz=50.0)             # 40–60 Hz 轻高通
        # 若需要极轻空间一致性，可指定统一的 IR 文件路径；不需要就保持 None
        REVERB_IR = None  # 例如 "irs/room_small.wav"
        out = _convolve_reverb(out, sr, ir_path=REVERB_IR, wet_db=-18.0)
        sf.write(out_wav, out, sr)
        return out_wav
# ---- XTTS (only if cached) --------------------------------------------------
def xtts_cached_model_path():
    # Linux 默认缓存位置；若你自定义了 TTS 缓存，可自行调整
    return Path.home() / ".local" / "share" / "tts" / "tts_models--multilingual--multi-dataset--xtts_v2" / "model.pth"

def xtts_available():
    return xtts_cached_model_path().exists()
class XTTSBackend:
    def __init__(self, device=None, language="zh-cn"):
        import importlib, subprocess as sp
        try:
            import TTS  # noqa
        except Exception:
            print("Installing TTS (XTTS backend) ...")
            sp.check_call([sys.executable, "-m", "pip", "install", "TTS==0.22.0"])
        from TTS.api import TTS as CoquiTTS
        self.tts = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try: self.tts.to(self.device)
        except Exception: pass

    def synth_sentencewise(self, ref_wav: Path, text: str, out_wav: Path, seed: int=0):
        import soundfile as sf
        torch.manual_seed(seed)
        sents = split_sentences(text)
        pieces=[]; sr=24000
        prev_tail_audio = None  # 新：保存上一段的尾巴音频用于对齐

        # 全局 LUFS 目标与 meter
        target_lufs = -23.0
        lufs_meter = None

        # —— 一次性抽 speaker embedding（conditioning_latents）——
        try:
            cond = self.tts.get_conditioning_latents(audio_path=str(ref_wav))
        except Exception:
            cond = None

        # 文本 look-back
        lookback_chars = 8
        prev_tail = ""

        for s in sents:
            context = prev_tail
            in_text = (context + s) if context else s
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            if cond is not None:
                # 直接用已缓存的 embedding
                self.tts.tts_to_file(
                    text=in_text, file_path=str(tmp_path),
                    speaker_wav=None, language=self.language,
                    speaker_embeddings=cond
                )
            else:
                # 回退：传原始参考
                self.tts.tts_to_file(
                    text=in_text, file_path=str(tmp_path),
                    speaker_wav=str(ref_wav), language=self.language
                )
            y, sr1 = sf.read(tmp_path); os.remove(tmp_path)
            if y.ndim>1: y=y.mean(axis=1)
            y = y.astype(np.float32)
            # if context:
            #     y = trim_head_by_chars(y, sr1, char_cnt=len(context))
            # y = seg_loudness_norm(y, -23.0)
            if context:
                y = trim_head_by_chars(y, sr1, char_cnt=len(context))
            if lufs_meter is None:
                lufs_meter = pyln.Meter(sr1)
            y = lufs_normalize(y, sr1, target_lufs, lufs_meter)

            # --- 新：缓存“上一段尾巴音频”用来做下一段对齐 ---
            tail_sec = _choose_tail_window_seconds(prev_tail, sr1)
            tail_len = min(len(y), int(tail_sec * sr1))
            prev_tail_audio = y[-tail_len:].copy()

            # 再做端点淡入淡出
            y = apply_fade(y, sr1)
            pieces.append(y); sr=sr1

            prev_tail = s[-lookback_chars:] if len(s) > lookback_chars else s

        out = concat_crossfade(pieces, sr, xfade_ms=60)
        out = _highpass(out, sr, cutoff_hz=50.0)             # 40–60 Hz 轻高通
        # 若需要极轻空间一致性，可指定统一的 IR 文件路径；不需要就保持 None
        REVERB_IR = None  # 例如 "irs/room_small.wav"
        out = _convolve_reverb(out, sr, ir_path=REVERB_IR, wet_db=-18.0)
        sf.write(out_wav, out, sr)
        return out_wav

# class XTTSBackend:
#     def __init__(self, device=None, language="zh-cn"):
#         # 延迟安装 TTS，仅在真的启用 XTTS 时引入，避免与 pandas 2.x 冲突
#         import importlib, subprocess as sp
#         try:
#             import TTS  # noqa
#         except Exception:
#             print("Installing TTS (XTTS backend) ...")
#             # 若你有严格版本需求，可固定: TTS==0.22.0
#             sp.check_call([sys.executable, "-m", "pip", "install", "TTS==0.22.0"])
#         from TTS.api import TTS as CoquiTTS
#         self.tts = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
#         self.language = language
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         try: self.tts.to(self.device)
#         except Exception: pass

#     def synth_sentencewise(self, ref_wav: Path, text: str, out_wav: Path, seed: int=0):
#         torch.manual_seed(seed)
#         sents = split_sentences(text)
#         pieces=[]; sr=24000
#         for s in sents:
#             with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
#                 tmp_path = Path(tmp.name)
#             self.tts.tts_to_file(text=s, file_path=str(tmp_path), speaker_wav=str(ref_wav), language=self.language)
#             y, sr1 = sf.read(tmp_path); os.remove(tmp_path)
#             if y.ndim>1: y=y.mean(axis=1)
#             pieces.append(apply_fade(y.astype(np.float32), sr1)); sr=sr1
#         out = concat_crossfade(pieces, sr, xfade_ms=10)
#         sf.write(out_wav, out, sr)
#         return out_wav

# ---- Piper (optional offline fallback) --------------------------------------
def piper_available(model_path: str):
    if not model_path: return False
    return Path(model_path).exists()

class PiperBackend:
    def __init__(self, model_path: str):
        import importlib, subprocess as sp
        try:
            import piper  # noqa
        except Exception:
            print("Installing piper-tts ...")
            sp.check_call([sys.executable, "-m", "pip", "install", "piper-tts"])
        from piper import PiperVoice
        self.voice = PiperVoice.load(model_path)

    def synth_sentencewise(self, text: str, out_wav: Path):
        sents = split_sentences(text)
        pieces=[]; sr=22050
        prev_tail_audio = None
        # 全局 LUFS 目标与 meter
        target_lufs = -23.0
        lufs_meter = None
        # 文本 look-back（Piper 没有参考，依旧使用上下文预热）
        lookback_chars = 8
        prev_tail = ""

        for s in sents:
            context = prev_tail
            in_text = (context + s) if context else s
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            self.voice.synthesize(in_text, str(tmp_path), sentence_silence=0.12)
            y, sr1 = sf.read(tmp_path); os.remove(tmp_path)
            if y.ndim>1: y=y.mean(axis=1)
            y = y.astype(np.float32)
            # if context:
            #     y = trim_head_by_chars(y, sr1, char_cnt=len(context))
            # y = seg_loudness_norm(y, -23.0)
            # y = apply_fade(y, sr1)
            if context:
                y = trim_head_by_chars(y, sr1, char_cnt=len(context))
            if lufs_meter is None:
                lufs_meter = pyln.Meter(sr1)
           
            y = lufs_normalize(y, sr1, target_lufs, lufs_meter)

            # --- 新：缓存“上一段尾巴音频”用来做下一段对齐 ---
            tail_sec = _choose_tail_window_seconds(prev_tail, sr1)
            tail_len = min(len(y), int(tail_sec * sr1))
            prev_tail_audio = y[-tail_len:].copy()

            # 再做端点淡入淡出
            y = apply_fade(y, sr1)

            pieces.append(y); sr=sr1

            prev_tail = s[-lookback_chars:] if len(s) > lookback_chars else s

        out = concat_crossfade(pieces, sr, xfade_ms=60)
        out = _highpass(out, sr, cutoff_hz=50.0)             # 40–60 Hz 轻高通
        # 若需要极轻空间一致性，可指定统一的 IR 文件路径；不需要就保持 None
        REVERB_IR = None  # 例如 "irs/room_small.wav"
        out = _convolve_reverb(out, sr, ir_path=REVERB_IR, wet_db=-18.0)
        sf.write(out_wav, out, sr)
        return out_wav

# class PiperBackend:
#     def __init__(self, model_path: str):
#         import importlib, subprocess as sp
#         try:
#             import piper  # noqa
#         except Exception:
#             print("Installing piper-tts ...")
#             sp.check_call([sys.executable, "-m", "pip", "install", "piper-tts"])
#         from piper import PiperVoice
#         self.voice = PiperVoice.load(model_path)

#     def synth_sentencewise(self, text: str, out_wav: Path):
#         sents = split_sentences(text)
#         pieces=[]; sr=22050
#         for s in sents:
#             with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
#                 tmp_path = Path(tmp.name)
#             # 句级直出
#             self.voice.synthesize(s, str(tmp_path), sentence_silence=0.12)
#             y, sr1 = sf.read(tmp_path); os.remove(tmp_path)
#             if y.ndim>1: y=y.mean(axis=1)
#             pieces.append(apply_fade(y.astype(np.float32), sr1)); sr=sr1
#         out = concat_crossfade(pieces, sr, xfade_ms=10)
#         sf.write(out_wav, out, sr)
#         return out_wav

# ------------------------ OFFLINE safety synths -------------------------------
def write_tone_placeholder(out_wav: Path, seconds=1.2, sr=16000, freq=440.0):
    t = np.linspace(0, seconds, int(sr*seconds), endpoint=False)
    y = 0.1*np.sin(2*np.pi*freq*t).astype(np.float32)
    y = apply_fade(y, sr, fade_ms=50)
    sf.write(out_wav, y, sr)
    return out_wav

def ref_mimic_collage(ref_wav: Path, target_seconds: float, out_wav: Path, sr=16000):
    y, sro = sf.read(ref_wav)
    if y.ndim>1: y=y.mean(axis=1)
    if sro != sr: y = librosa.resample(y, sro, sr)
    y = y.astype(np.float32)
    if len(y) < int(0.2*sr):
        return write_tone_placeholder(out_wav, seconds=max(0.6, target_seconds), sr=sr)

    frm = int(0.02*sr); hop=int(0.01*sr)
    frames = [y[i:i+frm] for i in range(0, max(1, len(y)-frm), hop)]
    rms = np.array([np.sqrt((f**2).mean()+1e-9) for f in frames])
    th = max(1e-4, np.median(rms)*0.6)
    voiced_idx = [i for i,r in enumerate(rms) if r>=th]
    if not voiced_idx:
        return write_tone_placeholder(out_wav, seconds=max(0.6, target_seconds), sr=sr)

    rng = random.Random(len(y))
    chunks=[]; total=0
    while total < int(target_seconds*sr):
        i = rng.choice(voiced_idx)
        s = max(0, i*hop - rng.randint(0, int(0.01*sr)))
        e = min(len(y), s + rng.randint(int(0.06*sr), int(0.16*sr)))
        seg = y[s:e].copy()
        seg = apply_fade(seg, sr, fade_ms=10)
        chunks.append(seg); total += len(seg)
        if len(chunks)>1000: break
    out = concat_crossfade(chunks, sr, xfade_ms=60)
    try:
        import scipy.signal as ss  # optional
        b,a = ss.butter(4, [70/(sr/2), 7000/(sr/2)], btype="band")
        out = ss.lfilter(b,a,out).astype(np.float32)
    except Exception:
        pass
    out = _highpass(out, sr, cutoff_hz=50.0)             # 40–60 Hz 轻高通
    # 若需要极轻空间一致性，可指定统一的 IR 文件路径；不需要就保持 None
    REVERB_IR = None  # 例如 "irs/room_small.wav"
    out = _convolve_reverb(out, sr, ir_path=REVERB_IR, wet_db=-18.0)
    sf.write(out_wav, out, sr)
    return out_wav

# ------------------------ selection ------------------------------------------
def pick_best(ref_wav: Path, text: str, cand_wavs: list, args, scorer=None,
              w_wer=0.4, w_sim=0.4, w_cm=0.2):
    if len(cand_wavs)==1 and args.no_asr and scorer is None:
        return cand_wavs[0], dict(wer=None, sim=None, cm=None)
    wers=[]; sims=[]; cms=[]
    for w in cand_wavs:
        try: werr,_ = asr_wer(w, text, args); wers.append(werr); print(f"\n{w}::scores:"); print("werr: ", werr)
        except Exception: wers.append(0.5)
        #try: sims.append(spk_similarity(ref_wav, w))
        try: sims_tmp = spk_similarity(ref_wav, w); sims.append(sims_tmp); print("sim:", sims_tmp)
        except Exception: sims.append(0.0)
        if scorer is not None:
            #try: cms.append(scorer.score(w))         
            try:
                cms_tmp = scorer.score(w)
                cms.append(cms_tmp)
                print("aasist:", cms_tmp)
            except Exception:
                cms.append(None)
        else:
            cms.append(None)
    min_w,max_w = min(wers), max(wers)
    min_s,max_s = min(sims), max(sims)
    valid_cms = [c for c in cms if c is not None]
    if valid_cms: min_c,max_c = min(valid_cms), max(valid_cms)
    scores=[]
    for i in range(len(cand_wavs)):
        wn = 0.0 if max_w==min_w else (wers[i]-min_w)/(max_w-min_w)
        sn = 0.0 if max_s==min_s else (sims[i]-min_s)/(max_s-min_s)
        if cms[i] is None or (valid_cms and max_c==min_c):
            final = 0.5*((1.0-wn)+sn)
        else:
            # 默认假设“分数越大越真” -> 惩罚(1-cn)就不合适；提供开关：
            #cn = (cms[i]-min_c)/(max_c-min_c) if max_c!=min_c else 0.5
            final = w_wer*(1.0-wn) + w_sim*sn + w_cm*0.5
        scores.append(final)
        print(f"{i}::final_score:"); print(final); print("\n") #####################
    best_idx = int(np.argmax(scores))
    return cand_wavs[best_idx], dict(wer=wers[best_idx], sim=sims[best_idx], cm=cms[best_idx])

# ------------------------ main -----------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    #ap.add_argument("--tasks_csv", default="aigc_speech_generation_tasks")
    ap.add_argument("--tasks_csv", default="tests2/aigc_speech_generation_tasks.csv")
    #ap.add_argument("--tasks_csv", default="aigc_speech_generation_tasks/aigc_speech_generation_tasks.csv")
    ap.add_argument("--team_name", default="result")
    ap.add_argument("--limit", type=int, default=0)

    # candidates
    ap.add_argument("--idx_model_dir", default="checkpoints")
    ap.add_argument("--idx_cfg", default="checkpoints/config.yaml")
    ap.add_argument("--k_idx", type=int, default=2)

    ap.add_argument("--use_xtts_if_cached", action="store_true")
    ap.add_argument("--k_xtts", type=int, default=1)

    ap.add_argument("--k_f5",  type=int, default=1)

    # ASR
    ap.add_argument("--no_asr", action="store_true")
    ap.add_argument("--asr_backend", choices=["faster","openai"], default="faster")
    ap.add_argument("--asr_device", choices=["auto","cpu","cuda"], default="auto")
    ap.add_argument("--asr_model", default="large-v3")
    ap.add_argument("--asr_download_root", default="models/whisper-large-v3")
    ap.add_argument("--asr_compute_type", default="float16")

    # AASIST
    ap.add_argument("--aasist_repo", default="")
    ap.add_argument("--aasist_ckpt", default="")
    # AASIST


    # >>> 新增：指定“另一个虚拟环境”的 python 与一个 CLI 脚本 <<<
    ap.add_argument("--python_aasist", default="", 
                    help="Path to python executable in the AASIST venv, e.g. /opt/venvs/aasist/bin/python")
    ap.add_argument("--aasist_cli", default="utils/aasist_score_cli.py",
                    help="Small CLI that returns anti-spoof score for a wav")


    # Piper (optional offline fallback)
    ap.add_argument("--use_piper_if_available", action="store_true")
    ap.add_argument("--piper_model", default="")  # e.g., models/piper/zh-CN.onnx

    args = ap.parse_args()

    df = pd.read_csv(args.tasks_csv)
    if args.limit>0: df = df.head(args.limit)

    from pathlib import Path

    out_dir = Path(args.team_name if args.team_name.endswith("-result") or args.team_name=="result" else f"{args.team_name}-result")
    out_dir.mkdir(parents=True, exist_ok=True)

    done_map = list_official_wavs(out_dir)      # {utt -> "result/utt.wav"}####################################
    done_set = set(done_map.keys())
    print(f"[resume] Found {len(done_set)} finished wav(s) in {out_dir}/ (skipping them).")

    # backends
    idx = None
    try:
        idx = IndexTTSBackend(model_dir=args.idx_model_dir, cfg_path=args.idx_cfg)
    except Exception as e:
        print("IndexTTS init failed, will rely on others:", e)

    xtts = None
    if args.use_xtts_if_cached and xtts_available():
        try:
            xtts = XTTSBackend()
            print("XTTS enabled from local cache.")
        except Exception as e:
            print("XTTS init failed, skip:", e)

    f5  = F5TTSCLIBackend()

    piper = None
    if args.use_piper_if_available and piper_available(args.piper_model):
        try:
            piper = PiperBackend(args.piper_model)
            print("Piper fallback enabled.")
        except Exception as e:
            print("Piper init failed, skip:", e)

    # # AASIST
    # scorer=None
    # if args.aasist_repo and args.aasist_ckpt:
    #     try:
    #         scorer = AASISTSpoofScorer(repo_dir=args.aasist_repo, ckpt_path=args.aasist_ckpt)
    #         print("AASIST enabled.")
    #     except Exception as e:
    #         print("AASIST init failed, continue without it:", e)

    # AASIST
    scorer = None

    if args.aasist_repo and args.aasist_ckpt:
        if args.python_aasist:
            from pathlib import Path
            if not Path(args.aasist_cli).exists():
                print(f"[AASIST] cli not found: {args.aasist_cli} -> disable AASIST")
            else:
                try:
                    scorer = ExternalAASISTScorer(
                        args.python_aasist, args.aasist_cli,
                        args.aasist_repo, args.aasist_ckpt
                    )
                    print(f"AASIST external scorer enabled via {args.python_aasist}")
                except Exception as e:
                    print("External AASIST init failed -> disable:", e)
                    scorer = None
        else:
            try:
                scorer = AASISTSpoofScorer(repo_dir=args.aasist_repo, ckpt_path=args.aasist_ckpt)
                print("AASIST in-process enabled.")
            except Exception as e:
                print("In-process AASIST init failed -> disable:", e)
                scorer = None

    



    # run_0817.py 中（开头）
#from ref_preproc import preprocess_reference, RefPreprocCfg

# …解析你的命令行参数后，构造配置（示例）
    cfg = RefPreprocCfg(
        target_sr=16000,
        do_denoise=True,        # 轻量降噪（避免把环境噪声当作说话人风格）
        vad_aggr=2,             # 2 较稳；若噪声多可试 3
        pick_seconds=6.0,       # 最终拼成 ~6秒参考
        max_seconds=60.0,       # 参考 wav 最长只看 1 分钟
    )

    # 在进入合成循环前/或每题目里，对当前参考语音做一次预处理并缓存
    # ref_in = f"./aigc_speech_generation_tasks/{row.reference_speech}"
    # ref_out = f".cache/refproc/{row.utt}_ref16k.wav"
    # _, sr, saved = preprocess_reference(ref_in, out_path=ref_out, cfg=cfg)

    # 之后把 ref_out 作为你 IndexTTS/F5/XTTS 的参考音频路径传入
    # e.g. f5-tts_infer-cli --ref_audio "{ref_out}" ...


    errors=[]; used_collage=0; used_tone=0; used_piper=0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Synthesize"):
        utt = int(row["utt"])
        if utt in done_set:    #######################################################
        # 已经有正式文件，直接跳过
            continue
        raw_text = str(row["text"])
        text = normalize_text(raw_text)
        ref_path = REF_DIR / str(row["reference_speech"])
        ref_path = Path(cut_reference(ref_path, max_sec=60, cache_dir=".cache/ref_trim"))
        out_final = out_dir / f"{utt}.wav"

        if out_final.exists(): continue

        if not ref_path.exists():
        #if not os.path.exists(ref_path):
            # 无参考 → 先试 Piper → 再拼贴 → 再哔声
            if piper is not None:
                try:
                    piper.synth_sentencewise(text, out_final); used_piper+=1
                    continue
                except Exception: pass
            try:
                _, sr, saved = preprocess_reference(ref_path, out_path=out_final, cfg=cfg)
                ref_mimic_collage(saved, estimate_duration(text), out_final); used_collage+=1
                #ref_mimic_collage(preprocess_ref(ref_path), estimate_duration(text), out_final); used_collage+=1
            except Exception:
                write_tone_placeholder(out_final, seconds=estimate_duration(text)); used_tone+=1
            continue

        # preprocess
        _, sr, saved = preprocess_reference(ref_path, out_path=out_final, cfg=cfg)
        ref = saved
        #ref = preprocess_ref(ref_path)

        # collect candidates (IndexTTS -> XTTS(opt) -> F5)
        cands=[]
        if idx is not None:
            for seed in range(args.k_idx):
                tmp = out_dir / f"_tmp_{utt}_idx_{seed}.wav"
                try:
                    idx.synth_sentencewise(ref, text, tmp, seed=seed); cands.append(tmp)
                except Exception as e:
                    errors.append((utt,"IndexTTS",repr(e)))

        if xtts is not None:
            for seed in range(args.k_xtts):
                tmp = out_dir / f"_tmp_{utt}_xtts_{seed}.wav"
                try:
                    xtts.synth_sentencewise(ref, text, tmp, seed=seed); cands.append(tmp)
                except Exception as e:
                    errors.append((utt,"XTTS",repr(e)))

        for k in range(args.k_f5):
            tmp = out_dir / f"_tmp_{utt}_f5_{k}.wav"
            try:
                f5.synth_sentencewise(ref, text, tmp); cands.append(tmp)
            except Exception as e:
                errors.append((utt,"F5TTS",repr(e)))

        if not cands:
            # 无候选 → Piper → 参考拼贴 → 哔声
            if piper is not None:
                try:
                    piper.synth_sentencewise(text, out_final); used_piper+=1
                    continue
                except Exception: pass
            try:
                ref_mimic_collage(ref, estimate_duration(text), out_final); used_collage+=1
            except Exception:
                write_tone_placeholder(out_final, seconds=estimate_duration(text)); used_tone+=1
            continue

        # pick best
        try:
            best, info = pick_best(ref, text, cands, args, scorer=scorer, w_wer=0.4, w_sim=0.4, w_cm=0.2)
        except Exception as e:
            errors.append((utt,"Scoring",repr(e)))
            best = cands[0]
        shutil.move(str(best), str(out_final))
        for w in cands:
            if w.exists():
                try: w.unlink()
                except: pass

        # light dither + fade
        try:
            y, sr = sf.read(out_final)
            noise = (np.random.randn(len(y))*1e-4).astype(np.float32)
            y = np.clip(y.astype(np.float32)+noise, -1.0, 1.0)
            y = apply_fade(y, sr, fade_ms=10)
            sf.write(out_final, y, sr)
        except Exception:
            pass

        gc.collect()

    # final sweep: ensure all wavs exist & non-trivial
    full = pd.read_csv(args.tasks_csv)
    fill_ct = 0
    for _, row in full.iterrows():
        utt = int(row["utt"]); out_wav = out_dir / f"{utt}.wav"
        if not out_wav.exists() or out_wav.stat().st_size <= 2048:
            try:
                rp = REF_DIR / str(row["reference_speech"])
                rp = Path(cut_reference(rp, max_sec=60, cache_dir=".cache/ref_trim"))
                if piper is not None:
                    piper.synth_sentencewise(str(row["text"]), out_wav); used_piper+=1; fill_ct+=1; continue
                if rp.exists():
                    ref_mimic_collage(preprocess_ref(rp), estimate_duration(str(row["text"])), out_wav); used_collage+=1; fill_ct+=1
                else:
                    write_tone_placeholder(out_wav, seconds=estimate_duration(str(row["text"]))); used_tone+=1; fill_ct+=1
            except Exception:
                write_tone_placeholder(out_wav, seconds=estimate_duration(str(row["text"]))); used_tone+=1; fill_ct+=1

    # CSV
    full["synthesized_speech"] = [f"{i}.wav" for i in full["utt"]]
    full.to_csv(out_dir / f"{out_dir.name}.csv", index=False)

    # ZIP
    zip_path = Path(f"{out_dir.name}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in out_dir.rglob("*"):
            zf.write(p, p.relative_to(out_dir.parent))

    print(f"\nDone. Dir: {out_dir}  Zip: {zip_path}")
    print(f"Used Piper: {used_piper}, Safety collage: {used_collage}, Tone placeholders: {used_tone}, Final sweep filled: {fill_ct}")
    if errors:
        print(f"Failures: {len(errors)} (first 5)")
        for e in errors[:5]: print("  ", e)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
