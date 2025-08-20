# scoring.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, csv, time
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import soundfile as sf
import librosa

# WER
try:
    from jiwer import wer as _jiwer
except Exception:
    _jiwer = None

# SpeechBrain ECAPA
try:
    from speechbrain.inference import EncoderClassifier
except Exception:
    from speechbrain.pretrained import EncoderClassifier

import torch

# ====== 新增：AASIST 封装（可选） ======
try:
    from aasist_wrapper import AASISTScorer
except Exception:
    AASISTScorer = None  # type: ignore


class CandidateScorer:
    def __init__(
        self,
        asr_backend: str = "faster",
        asr_model: str = "large-v3",
        asr_device: str = "auto",
        asr_compute_type: str = "float16",
        asr_download_root: str = "models/whisper-large-v3",
        no_asr: bool = False,
        w_wer: float = 0.4,
        w_sim: float = 0.4,
        # ====== 新增：AASIST 参数 ======
        use_aasist: bool = False,
        w_aasist: float = 0.2,
        aasist_device: str = "auto",
        log_path: str | Path = "logs/score_log.csv",
    ):
        self.asr_backend = asr_backend
        self.asr_model = asr_model
        self.asr_device = asr_device
        self.asr_compute_type = asr_compute_type
        self.asr_download_root = asr_download_root
        self.no_asr = no_asr

        self.w_wer = float(w_wer)
        self.w_sim = float(w_sim)

        # ====== 新增：AASIST ======
        self.use_aasist = bool(use_aasist)
        self.w_aasist = float(w_aasist)
        self._aasist = None
        if self.use_aasist and AASISTScorer is not None:
            try:
                self._aasist = AASISTScorer(device=aasist_device)
                if not self._aasist.available():
                    print("[scoring] AASIST not available, disable it.")
                    self._aasist = None
            except Exception as e:
                print("[scoring] AASIST init failed:", e)
                self._aasist = None

        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_log_header()

        # lazy init
        self._faster_model = None
        self._openai_whisper = None
        self._spk = None
        print(f"[scoring] score log -> {self.log_path.resolve()}")


    # ---------- logging ----------
    def _ensure_log_header(self):
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "ts", "utt", "tag", "wav_path",
                    "wer", "sim", "aasist",              # 新增原始列
                    "wer_norm", "sim_norm", "aasist_norm",
                    "final_score", "chosen"
                ])

    def _log_row(self, utt: str | int, tag: str, wav_path: Path,
                 wer: Optional[float], sim: Optional[float], aas: Optional[float],
                 wer_norm: Optional[float], sim_norm: Optional[float], aas_norm: Optional[float],
                 final_score: Optional[float], chosen: bool):
        with self.log_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                int(time.time()),
                str(utt),
                tag,
                str(wav_path),
                "" if wer is None else f"{wer:.6f}",
                "" if sim is None else f"{sim:.6f}",
                "" if aas is None else f"{aas:.6f}",
                "" if wer_norm is None else f"{wer_norm:.6f}",
                "" if sim_norm is None else f"{sim_norm:.6f}",
                "" if aas_norm is None else f"{aas_norm:.6f}",
                "" if final_score is None else f"{final_score:.6f}",
                "1" if chosen else "0"
            ])

    # ---------- utils ----------
    def _device(self) -> str:
        if self.asr_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.asr_device

    # ---------- ASR ----------
    def _ensure_faster(self):
        if self._faster_model is not None:
            return
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception:
            import subprocess
            print("[scoring] Installing faster-whisper ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "faster-whisper", "huggingface_hub"])
            from faster_whisper import WhisperModel  # type: ignore
        self._faster_model = WhisperModel(
            self.asr_model,
            device=self._device(),
            compute_type=self.asr_compute_type,
            download_root=self.asr_download_root,
        )

    # def _ensure_openai_whisper(self):
    #     if self._openai_whisper is not None:
    #         return
    #     try:
    #         import whisper as oaiw  # type: ignore
    #     except Exception:
    #         import subprocess
    #         print("[scoring] Installing openai-whisper ...")
    #         subprocess.check_call([sys.executable, "-m", "pip", "install", "openai-whisper"])
    #         import whisper as oaiw  # type: ignore
    #     self._openai_whisper = oaiw.load_model(self.asr_model, device=self._device())

    def _ensure_openai_whisper(self):
        if self._openai_whisper is not None:
            return
        import subprocess, importlib
        try:
            import whisper as oaiw  # 可能是错包
            if not hasattr(oaiw, "load_model"):
                raise ImportError("wrong whisper package")
        except Exception:
            # 卸载错包，安装对的 openai-whisper
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "whisper"])
            except Exception:
                pass
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "openai-whisper"])
            import whisper as oaiw
            if not hasattr(oaiw, "load_model"):
                raise RuntimeError("openai-whisper installed but `whisper.load_model` still missing")
        self._openai_whisper = oaiw.load_model(self.asr_model, device=self._device())


    def _asr_transcribe(self, wav_path: Path) -> str:
        if self.no_asr or _jiwer is None:
            return ""
        if self.asr_backend == "faster":
            self._ensure_faster()
            segs, info = self._faster_model.transcribe(str(wav_path), language="zh", beam_size=5)
            return "".join([s.text for s in segs]).strip()
        else:
            self._ensure_openai_whisper()
            res = self._openai_whisper.transcribe(str(wav_path), language="zh")
            return res.get("text", "").strip()

    # ---------- SIM ----------
    def _ensure_spk(self):
        if self._spk is not None:
            return
        self._spk = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self._device()}
        )

    def _embed(self, wav_path: Path) -> np.ndarray:
        sig, sr = sf.read(wav_path)
        if sig.ndim > 1:
            sig = sig.mean(axis=1)
        if sr != 16000:
            sig = librosa.resample(sig, sr, 16000)
        t = torch.from_numpy(sig).float().unsqueeze(0)
        with torch.no_grad():
            e = self._spk.encode_batch(t).squeeze(0).squeeze(0).cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    def _sim(self, ref_wav: Path, syn_wav: Path) -> float:
        self._ensure_spk()
        a = self._embed(ref_wav)
        b = self._embed(syn_wav)
        return float(np.dot(a, b))

    # ---------- AASIST ----------
    def _aasist_bonafide(self, wav_path: Path) -> Optional[float]:
        if self._aasist is None:
            return None
        try:
            return self._aasist.score_bonafide(str(wav_path))
        except Exception:
            return None

    # ---------- public API ----------
    def score_candidates(
        self,
        ref_embed_wav: Path,
        ref_text: str,
        candidates: List[Tuple[Path, str]],   # (wav_path, tag)
        utt: str | int,
    ) -> Tuple[Path, Dict]:
        if not candidates:
            raise ValueError("No candidates to score.")

        wers: List[Optional[float]] = []
        sims: List[Optional[float]] = []
        aasist_vals: List[Optional[float]] = []

        # 逐候选计算原始指标
        for wav_path, tag in candidates:
            # WER
            if self.no_asr or _jiwer is None:
                werr = None
            else:
                hyp = ""
                try:
                    hyp = self._asr_transcribe(wav_path)
                except Exception:
                    hyp = ""
                if hyp:
                    try:
                        werr = float(_jiwer(ref_text.strip(), hyp))
                    except Exception:
                        werr = None
                else:
                    werr = None
            wers.append(werr)

            # SIM
            try:
                simv = self._sim(ref_embed_wav, wav_path)
            except Exception:
                simv = None
            sims.append(simv)

            # AASIST
            if self._aasist is not None:
                aas = self._aasist_bonafide(wav_path)
            else:
                aas = None
            aasist_vals.append(aas)

        # 归一化（忽略 None）
        def _minmax(vs):
            vals = [v for v in vs if v is not None]
            if not vals:
                return (0.0, 0.0)
            return (min(vals), max(vals))

        min_w, max_w = _minmax(wers)
        min_s, max_s = _minmax(sims)
        min_a, max_a = _minmax(aasist_vals)

        wn_list: List[Optional[float]] = []
        sn_list: List[Optional[float]] = []
        an_list: List[Optional[float]] = []
        scores: List[Optional[float]] = []

        for i, (wav_path, tag) in enumerate(candidates):
            werr = wers[i]
            simv = sims[i]
            aas  = aasist_vals[i]

            # wer_norm：越小越好
            if werr is None or max_w == min_w:
                wer_norm = None
            else:
                wer_norm = float(np.clip((werr - min_w) / (max_w - min_w), 0.0, 1.0))
            # sim_norm：越大越好
            if simv is None or max_s == min_s:
                sim_norm = None
            else:
                sim_norm = float(np.clip((simv - min_s) / (max_s - min_s), 0.0, 1.0))
            # aasist_norm：越大越好
            if aas is None or max_a == min_a:
                aas_norm = None
            else:
                aas_norm = float(np.clip((aas - min_a) / (max_a - min_a), 0.0, 1.0))

            # 动态权重：缺失的项不计入
            ww = 0.0 if wer_norm is None else self.w_wer
            ws = 0.0 if sim_norm is None else self.w_sim
            wa = 0.0 if (aas_norm is None or self._aasist is None) else self.w_aasist
            denom = ww + ws + wa

            if denom == 0.0:
                final = None
            else:
                final = (
                    ww * (1.0 - (wer_norm if wer_norm is not None else 0.0)) +
                    ws * (sim_norm if sim_norm is not None else 0.0) +
                    wa * (aas_norm if aas_norm is not None else 0.0)
                ) / denom

            wn_list.append(wer_norm)
            sn_list.append(sim_norm)
            an_list.append(aas_norm)
            scores.append(final)

        # 选择最佳
        best_idx = None
        if any(s is not None for s in scores):
            best_idx = int(np.nanargmax([(-1) if s is None else s for s in scores]))
        elif any(s is not None for s in sims):
            best_idx = int(np.nanargmax([(-1) if s is None else s for s in sims]))
        else:
            best_idx = 0

        # 写日志：逐条
        for i, (wav_path, tag) in enumerate(candidates):
            self._log_row(
                utt=utt,
                tag=tag,
                wav_path=wav_path,
                wer=wers[i],
                sim=sims[i],
                aas=aasist_vals[i],
                wer_norm=wn_list[i],
                sim_norm=sn_list[i],
                aas_norm=an_list[i],
                final_score=scores[i],
                chosen=(i == best_idx),
            )

        best_wav = candidates[best_idx][0]
        return best_wav, {"wer": wers[best_idx], "sim": sims[best_idx],
                          "aasist": aasist_vals[best_idx]}
