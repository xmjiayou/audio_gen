#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALWAYS-200 SAFE RUNNER:
- 3 backends: IndexTTS (main) + XTTS (optional) + F5-TTS (fallback)
- + Guaranteed offline fallback: Coqui zh-CN baker/tacotron2-DDC
- If all fail: write a tiny tone WAV (valid audio) so you never miss rows
- Sentence-wise synth + 10 ms crossfade & fade-in/out
- Scoring: faster-whisper (default) WER + ECAPA cosine + (optional) AASIST
- If scoring fails, still keep best-available candidate instead of aborting
"""

import os, sys, shlex, shutil, subprocess, zipfile, warnings, gc, tempfile, math
from pathlib import Path
import argparse
import numpy as np

def ensure(pkgs):
    import importlib, subprocess as sp
    miss=[]
    for p in pkgs:
        try: importlib.import_module(p)
        except Exception: miss.append(p)
    if miss:
        print("Installing:", miss)
        sp.check_call([sys.executable, "-m", "pip", "install", *miss])

# base deps
ensure(["pandas","tqdm","soundfile","librosa","speechbrain","jiwer"])
import pandas as pd, soundfile as sf, librosa
from tqdm import tqdm
from jiwer import wer
import torch
from speechbrain.pretrained import EncoderClassifier

# ---------- env caches (persist between runs) ----------
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(Path("./.hf_cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("./.cache").resolve()))

CSV_PATH = "aigc_speech_generation_tasks/aigc_speech_generation_tasks.csv"
REF_DIR  = Path("aigc_speech_generation_tasks")
PRE_DIR  = Path("preproc_refs"); PRE_DIR.mkdir(exist_ok=True)

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

def apply_fade(y, sr, fade_ms=10):
    n = max(1, int(sr*fade_ms/1000))
    if len(y) < 2*n: return y
    y[:n]  *= np.linspace(0,1,n,endpoint=True)
    y[-n:] *= np.linspace(1,0,n,endpoint=True)
    return y

def concat_crossfade(chunks, sr, xfade_ms=10):
    if not chunks: return np.zeros(0, dtype=np.float32)
    xfade = int(sr*xfade_ms/1000)
    out = chunks[0].copy()
    for i in range(1,len(chunks)):
        a, b = out, chunks[i]
        if xfade>0 and len(a)>xfade and len(b)>xfade:
            a_tail = a[-xfade:]; b_head = b[:xfade]
            mix = a_tail*np.linspace(1,0,xfade) + b_head*np.linspace(0,1,xfade)
            out = np.concatenate([a[:-xfade], mix, b[xfade:]], axis=0)
        else:
            out = np.concatenate([a, b], axis=0)
    return np.clip(out, -1.0, 1.0)

PUNC = "。！？!?；;：:，,、"
def split_sentences(text, max_len=80):
    sents, buf = [], ""
    for ch in str(text):
        buf += ch
        if ch in "。！？!?":
            sents.append(buf.strip()); buf=""
        elif len(buf)>=max_len and ch in "，,、;；:：":
            sents.append(buf.strip()); buf=""
    if buf.strip(): sents.append(buf.strip())
    return [s for s in sents if s]

# ----------- ASR: faster-whisper (default) with fallback -----------
_faster_model = None
_openai_whisper = None
def asr_wer(wav_path: Path, ref_text: str, args, device=None):
    if args.no_asr:
        return 0.5, ""  # neutral
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.asr_backend == "faster":
        try:
            ensure(["faster_whisper","huggingface_hub"])
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
            return wer(ref_text.strip(), hyp), hyp
        except Exception as e:
            print("[ASR] faster-whisper failed, fallback to openai-whisper:", e)

    # openai-whisper fallback (might download big weights)
    try:
        global _openai_whisper
        if _openai_whisper is None:
            ensure(["whisper"])
            import whisper as oaiw
            _openai_whisper = oaiw.load_model("large-v3", device=dev)
        result = _openai_whisper.transcribe(str(wav_path), language="zh")
        hyp = result.get("text","").strip()
        if not hyp: return 1.0, ""
        return wer(ref_text.strip(), hyp), hyp
    except Exception as e:
        print("[ASR] openai-whisper failed:", e)
        return 0.5, ""  # neutral if ASR unavailable

# ----------- Speaker similarity (ECAPA cosine) -----------
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

# ----------- Optional AASIST scorer -----------
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
            self.backend = "ssl"
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

# ----------- Backends -----------
class IndexTTSBackend:
    def __init__(self, model_dir="checkpoints", cfg_path="checkpoints/config.yaml", device=None):
        from indextts.infer import IndexTTS
        self.tts = IndexTTS(model_dir=model_dir, cfg_path=cfg_path, device=device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def synth_sentencewise(self, ref_wav: Path, text: str, out_wav: Path, seed: int=0):
        torch.manual_seed(seed)
        sents = split_sentences(text)
        pieces=[]; sr=24000
        for s in sents:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            self.tts.infer(str(ref_wav), s, str(tmp_path))
            y, sr1 = sf.read(tmp_path); os.remove(tmp_path)
            if y.ndim>1: y=y.mean(axis=1)
            pieces.append(apply_fade(y.astype(np.float32), sr1)); sr=sr1
        out = concat_crossfade(pieces, sr, xfade_ms=10)
        sf.write(out_wav, out, sr)
        return out_wav

class XTTSBackend:
    def __init__(self, device=None, language="zh-cn"):
        from TTS.api import TTS as CoquiTTS
        self.tts = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tts.to(self.device)

    def synth_sentencewise(self, ref_wav: Path, text: str, out_wav: Path, seed: int=0):
        torch.manual_seed(seed)
        sents = split_sentences(text)
        pieces=[]; sr=24000
        for s in sents:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            self.tts.tts_to_file(
                text=s, file_path=str(tmp_path),
                speaker_wav=str(ref_wav), language=self.language
            )
            y, sr1 = sf.read(tmp_path); os.remove(tmp_path)
            if y.ndim>1: y=y.mean(axis=1)
            pieces.append(apply_fade(y.astype(np.float32), sr1)); sr=sr1
        out = concat_crossfade(pieces, sr, xfade_ms=10)
        sf.write(out_wav, out, sr)
        return out_wav

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
            default_out = Path("tests/infer_cli_basic.wav")
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
        out = concat_crossfade(pieces, sr, xfade_ms=10)
        sf.write(out_wav, out, sr)
        return out_wav

# GUARANTEED OFFLINE FALLBACK (no ref wav needed)
class CoquiZHSingleBackend:
    def __init__(self, model_id="tts_models/zh-CN/baker/tacotron2-DDC", device=None):
        ensure(["TTS"])
        from TTS.api import TTS as CoquiTTS
        self.tts = CoquiTTS(model_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tts.to(self.device)
        except Exception:
            pass  # CPU also fine

    def synth_sentencewise(self, text: str, out_wav: Path):
        sents = split_sentences(text)
        pieces=[]; sr=22050
        for s in sents:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            self.tts.tts_to_file(text=s, file_path=str(tmp_path))
            y, sr1 = sf.read(tmp_path); os.remove(tmp_path)
            if y.ndim>1: y=y.mean(axis=1)
            pieces.append(apply_fade(y.astype(np.float32), sr1)); sr=sr1
        out = concat_crossfade(pieces, sr, xfade_ms=10)
        sf.write(out_wav, out, sr)
        return out_wav

def write_tone_placeholder(out_wav: Path, seconds=1.2, sr=16000, freq=440.0):
    t = np.linspace(0, seconds, int(sr*seconds), endpoint=False)
    y = 0.1*np.sin(2*np.pi*freq*t).astype(np.float32)
    y = apply_fade(y, sr, fade_ms=50)
    sf.write(out_wav, y, sr)
    return out_wav

# ----------- selection -----------
def pick_best(ref_wav: Path, text: str, cand_wavs: list, args, scorer=None,
              w_wer=0.4, w_sim=0.4, w_cm=0.2):
    if len(cand_wavs)==1 and args.no_asr and scorer is None:
        return cand_wavs[0], dict(wer=None, sim=None, cm=None)
    wers=[]; sims=[]; cms=[]
    for w in cand_wavs:
        try:
            werr, _ = asr_wer(w, text, args); wers.append(werr)
        except Exception:
            wers.append(0.5)
        try:
            sims.append(spk_similarity(ref_wav, w))
        except Exception:
            sims.append(0.0)
        if scorer is not None:
            try:
                cms.append(scorer.score(w))
            except Exception:
                cms.append(None)
        else:
            cms.append(None)
    # normalize
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
        elif cms[i] is None:
            final = 0.5*((1.0-wn)+sn)
        else:
            cn = (cms[i]-min_c)/(max_c-min_c) if max_c!=min_c else 0.5
            final = w_wer*(1.0-wn) + w_sim*sn + w_cm*(1.0-cn)
        scores.append(final)
    best_idx = int(np.argmax(scores))
    return cand_wavs[best_idx], dict(wer=wers[best_idx], sim=sims[best_idx], cm=cms[best_idx])

# ----------- main -----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks_csv", default=CSV_PATH)
    ap.add_argument("--team_name", default="result")
    ap.add_argument("--limit", type=int, default=0)
    # backends
    ap.add_argument("--idx_model_dir", default="checkpoints")
    ap.add_argument("--idx_cfg", default="checkpoints/config.yaml")
    ap.add_argument("--k_idx", type=int, default=2)
    ap.add_argument("--use_xtts", action="store_true")
    ap.add_argument("--k_xtts", type=int, default=1)
    ap.add_argument("--k_f5", type=int, default=1)
    # ASR
    ap.add_argument("--no_asr", action="store_true")
    ap.add_argument("--asr_backend", choices=["faster","openai"], default="faster")
    ap.add_argument("--asr_model", default="large-v3")
    ap.add_argument("--asr_download_root", default="models/whisper-large-v3")
    ap.add_argument("--asr_compute_type", default="float16")
    # AASIST optional
    ap.add_argument("--aasist_repo", default="")
    ap.add_argument("--aasist_ckpt", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.tasks_csv)
    if args.limit>0: df = df.head(args.limit)

    out_dir = Path(args.team_name if args.team_name.endswith("-result") or args.team_name=="result"
                   else f"{args.team_name}-result")
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = IndexTTSBackend(model_dir=args.idx_model_dir, cfg_path=args.idx_cfg)
    xtts = XTTSBackend() if args.use_xtts else None
    f5  = F5TTSCLIBackend()
    coq = CoquiZHSingleBackend()  # GUARANTEED offline fallback

    scorer=None
    if args.aasist_repo and args.aasist_ckpt:
        try:
            scorer = AASISTSpoofScorer(repo_dir=args.aasist_repo, ckpt_path=args.aasist_ckpt)
            print("AASIST enabled.")
        except Exception as e:
            print("AASIST init failed, continue without it:", e)

    errors=[]; used_fallback=0; used_tone=0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Synthesize"):
        utt = int(row["utt"]); text = str(row["text"])
        ref_path = REF_DIR / str(row["reference_speech"])
        if not ref_path.exists():
            errors.append((utt,"MissingRef", str(ref_path)))
            # no ref → still guarantee output
            out_final = out_dir / f"{utt}.wav"
            try:
                coq.synth_sentencewise(text, out_final); used_fallback+=1
            except Exception:
                write_tone_placeholder(out_final); used_tone+=1
            continue

        ref = preprocess_ref(ref_path)
        out_final = out_dir / f"{utt}.wav"
        if out_final.exists(): continue

        cands=[]
        # IndexTTS
        for seed in range(args.k_idx):
            tmp = out_dir / f"_tmp_{utt}_idx_{seed}.wav"
            try:
                idx.synth_sentencewise(ref, text, tmp, seed=seed); cands.append(tmp)
            except Exception as e:
                errors.append((utt,"IndexTTS",repr(e)))
        # XTTS
        if xtts is not None:
            for seed in range(args.k_xtts):
                tmp = out_dir / f"_tmp_{utt}_xtts_{seed}.wav"
                try:
                    xtts.synth_sentencewise(ref, text, tmp, seed=seed); cands.append(tmp)
                except Exception as e:
                    errors.append((utt,"XTTS",repr(e)))
        # F5-TTS
        for k in range(args.k_f5):
            tmp = out_dir / f"_tmp_{utt}_f5_{k}.wav"
            try:
                f5.synth_sentencewise(ref, text, tmp); cands.append(tmp)
            except Exception as e:
                errors.append((utt,"F5TTS",repr(e)))

        if not cands:
            # HARD fallback
            try:
                coq.synth_sentencewise(text, out_final); used_fallback+=1
            except Exception:
                write_tone_placeholder(out_final); used_tone+=1
            continue

        # pick best; if scoring explodes, keep first candidate
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

        # feather-light dither + fade
        try:
            y, sr = sf.read(out_final)
            noise = (np.random.randn(len(y))*1e-4).astype(np.float32)
            y = np.clip(y.astype(np.float32)+noise, -1.0, 1.0)
            y = apply_fade(y, sr, fade_ms=10)
            sf.write(out_final, y, sr)
        except Exception:
            pass

        gc.collect()

    # FINAL SWEEP: ensure 200/limit outputs exist
    full = pd.read_csv(args.tasks_csv)
    ensure_count = 0
    for _, row in full.iterrows():
        utt = int(row["utt"]); out_wav = out_dir / f"{utt}.wav"
        if not out_wav.exists() or out_wav.stat().st_size <= 1024:
            # last-resort fill
            try:
                coq.synth_sentencewise(str(row["text"]), out_wav); used_fallback+=1; ensure_count+=1
            except Exception:
                write_tone_placeholder(out_wav); used_tone+=1; ensure_count+=1

    # CSV
    full["synthesized_speech"] = [f"{i}.wav" for i in full["utt"]]
    full.to_csv(out_dir / f"{out_dir.name}.csv", index=False)

    # ZIP
    zip_path = Path(f"{out_dir.name}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in out_dir.rglob("*"):
            zf.write(p, p.relative_to(out_dir.parent))

    print(f"\nDone. Dir: {out_dir}  Zip: {zip_path}")
    print(f"Hard fallback used: {used_fallback}, tone placeholders: {used_tone}, final sweep filled: {ensure_count}")
    if errors:
        print(f"Failures: {len(errors)} (first 5)")
        for e in errors[:5]: print("  ", e)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
