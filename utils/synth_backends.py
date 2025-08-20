# synth_backends.py
# -*- coding: utf-8 -*-
"""
TTS 合成后端的统一封装：
- IndexTTSBackend（IndexTTS）
- XTTSBackend（仅当本地缓存存在时启用）
- F5TTSCLIBackend（f5-tts 命令行）
- PiperBackend（离线 piper-tts）
并提供可用性探测：xtts_available(), piper_available()

注意：这里只做“模型使用部分”的封装；文本清洗、参考音频预处理、ASR 打分等留在主脚本。
"""
from __future__ import annotations
import os, shutil, subprocess, tempfile, shlex
from pathlib import Path
from typing import Optional, List

import numpy as np
import soundfile as sf

# ---------------------- small audio helpers (局部复用) ----------------------
def _apply_fade(y: np.ndarray, sr: int, fade_ms: int = 10) -> np.ndarray:
    n = max(1, int(sr * fade_ms / 1000))
    if len(y) < 2 * n:
        return y
    y = y.astype(np.float32, copy=True)
    y[:n] *= np.linspace(0, 1, n, endpoint=True).astype(np.float32)
    y[-n:] *= np.linspace(1, 0, n, endpoint=True).astype(np.float32)
    return y

def _concat_crossfade(chunks: List[np.ndarray], sr: int, xfade_ms: int = 10) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    xfade = int(sr * xfade_ms / 1000)
    out = chunks[0].astype(np.float32, copy=True)
    for i in range(1, len(chunks)):
        a, b = out, chunks[i].astype(np.float32, copy=False)
        if xfade > 0 and len(a) > xfade and len(b) > xfade:
            a_tail = a[-xfade:]
            b_head = b[:xfade]
            alpha = np.linspace(0, 1, xfade, endpoint=True).astype(np.float32)
            mix = a_tail * (1 - alpha) + b_head * alpha
            out = np.concatenate([a[:-xfade], mix, b[xfade:]], axis=0)
        else:
            out = np.concatenate([a, b], axis=0)
    return np.clip(out, -1.0, 1.0)

def _split_sentences(text: str, max_len: int = 80) -> list[str]:
    PUNC_HARD = "。！？!?"
    PUNC_SOFT = "，,、;；:："
    sents, buf = [], ""
    for ch in str(text):
        buf += ch
        if ch in PUNC_HARD:
            sents.append(buf.strip()); buf = ""
        elif len(buf) >= max_len and ch in PUNC_SOFT:
            sents.append(buf.strip()); buf = ""
    if buf.strip():
        sents.append(buf.strip())
    return [s for s in sents if s]

# ---------------------- availability probes ----------------------
def xtts_cached_model_path() -> Path:
    return Path.home() / ".local" / "share" / "tts" / \
           "tts_models--multilingual--multi-dataset--xtts_v2" / "model.pth"

def xtts_available() -> bool:
    return xtts_cached_model_path().exists()

def piper_available(model_path: str | Path) -> bool:
    return bool(model_path) and Path(model_path).exists()

# ---------------------- Backends ----------------------
class IndexTTSBackend:
    """
    IndexTTS 推理封装
    需要：from indextts.infer import IndexTTS
    """
    def __init__(self, model_dir="checkpoints", cfg_path="checkpoints/config.yaml", device: Optional[str] = None):
        from indextts.infer import IndexTTS
        self.tts = IndexTTS(model_dir=model_dir, cfg_path=cfg_path, device=device)

    def synth_sentencewise(self, ref_wav: Path, text: str, out_wav: Path, seed: int = 0,
                           fade_ms: int = 10, xfade_ms: int = 10) -> Path:
        import torch
        torch.manual_seed(seed)

        sents = _split_sentences(text)
        pieces = []
        sr_out = 24000
        for s in sents:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            self.tts.infer(str(ref_wav), s, str(tmp_path))
            y, sr1 = sf.read(tmp_path); os.remove(tmp_path)
            if y.ndim > 1: y = y.mean(axis=1)
            y = _apply_fade(y.astype(np.float32), sr1, fade_ms=fade_ms)
            pieces.append(y); sr_out = sr1
        mixed = _concat_crossfade(pieces, sr_out, xfade_ms=xfade_ms)
        sf.write(out_wav, mixed, sr_out)
        return out_wav

class XTTSBackend:
    """
    XTTS v2（Coqui TTS 0.22.0）封装
    仅当本地缓存存在（离线环境）时再初始化
    """
    def __init__(self, device: Optional[str] = None, language: str = "zh-cn"):
        # 不在此处安装依赖，缺失直接抛错，交给上层决定是否跳过
        from TTS.api import TTS as CoquiTTS  # TTS==0.22.0
        if not xtts_available():
            raise FileNotFoundError("XTTS cache not found at ~/.local/share/tts/.../xtts_v2/")
        self.tts = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
        self.language = language
        self.device = device
        try:
            self.tts.to(device or "cuda")
        except Exception:
            pass

    def synth_sentencewise(self, ref_wav: Path, text: str, out_wav: Path, seed: int = 0,
                           fade_ms: int = 10, xfade_ms: int = 10) -> Path:
        import torch
        torch.manual_seed(seed)

        sents = _split_sentences(text)
        pieces = []
        sr_out = 24000
        for s in sents:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            self.tts.tts_to_file(text=s, file_path=str(tmp_path),
                                 speaker_wav=str(ref_wav), language=self.language)
            y, sr1 = sf.read(tmp_path); os.remove(tmp_path)
            if y.ndim > 1: y = y.mean(axis=1)
            y = _apply_fade(y.astype(np.float32), sr1, fade_ms=fade_ms)
            pieces.append(y); sr_out = sr1
        mixed = _concat_crossfade(pieces, sr_out, xfade_ms=xfade_ms)
        sf.write(out_wav, mixed, sr_out)
        return out_wav

class F5TTSCLIBackend:
    """
    f5-tts 命令行封装（无代码 API 时的简单适配）
    """
    def __init__(self, model: str = "F5TTS_v1_Base"):
        self.model = model
        # 交给外部准备依赖；也可以这里尝试安装：
        if shutil.which("f5-tts_infer-cli") is None:
            try:
                subprocess.check_call([os.sys.executable, "-m", "pip", "install", "f5-tts"])
            except Exception:
                pass  # 让上层捕获失败

    def synth_sentencewise(self, ref_wav: Path, text: str, out_wav: Path,
                           fade_ms: int = 10, xfade_ms: int = 10) -> Path:
        sents = _split_sentences(text)
        tmp_wavs = []
        for i, s in enumerate(sents):
            cmd = (
                f'f5-tts_infer-cli --model {self.model} '
                f'--ref_audio {shlex.quote(str(ref_wav))} --gen_text {shlex.quote(s)}'
            )
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            default_out = Path("tests/infer_cli_basic.wav")
            if not default_out.exists():
                raise FileNotFoundError("F5-TTS output not found: tests/infer_cli_basic.wav")
            p = out_wav.parent / f"_tmp_f5_{out_wav.stem}_{i}.wav"
            shutil.move(str(default_out), str(p))
            tmp_wavs.append(p)

        pieces = []; sr_out = 24000
        for p in tmp_wavs:
            y, sr1 = sf.read(p); p.unlink(missing_ok=True)
            if y.ndim > 1: y = y.mean(axis=1)
            y = _apply_fade(y.astype(np.float32), sr1, fade_ms=fade_ms)
            pieces.append(y); sr_out = sr1
        mixed = _concat_crossfade(pieces, sr_out, xfade_ms=xfade_ms)
        sf.write(out_wav, mixed, sr_out)
        return out_wav

class PiperBackend:
    """
    piper-tts（离线）封装
    """
    def __init__(self, model_path: str | Path):
        from piper import PiperVoice
        self.voice = PiperVoice.load(str(model_path))

    def synth_sentencewise(self, text: str, out_wav: Path,
                           sentence_silence: float = 0.12,
                           fade_ms: int = 10, xfade_ms: int = 10) -> Path:
        sents = _split_sentences(text)
        pieces = []
        sr_out = 22050
        for s in sents:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            self.voice.synthesize(s, str(tmp_path), sentence_silence=sentence_silence)
            y, sr1 = sf.read(tmp_path); os.remove(tmp_path)
            if y.ndim > 1: y = y.mean(axis=1)
            y = _apply_fade(y.astype(np.float32), sr1, fade_ms=fade_ms)
            pieces.append(y); sr_out = sr1
        mixed = _concat_crossfade(pieces, sr_out, xfade_ms=xfade_ms)
        sf.write(out_wav, mixed, sr_out)
        return out_wav
