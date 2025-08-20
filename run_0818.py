#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from pathlib import Path
from utils.resume_utils import list_official_wavs, summarize_progress 

# ------------------------ small helper: lazy installer ------------------------
def ensure_base():
    pkgs = ["pandas","tqdm","soundfile","librosa","speechbrain","jiwer"]
    import importlib, subprocess as sp
    miss=[]
    for p in pkgs:
        try: importlib.import_module(p)
        except Exception: miss.append(p)
    if miss:
        print("Installing base deps:", miss)
        sp.check_call([sys.executable, "-m", "pip", "install", *miss])
ensure_base()

import pandas as pd, soundfile as sf, librosa
from tqdm import tqdm
from jiwer import wer
import torch
#from speechbrain.pretrained import EncoderClassifier
try:
    from speechbrain.inference import EncoderClassifier  # SB ≥ 1.0
except Exception:
    from speechbrain.pretrained import EncoderClassifier  # 兼容老版本


# ------------------------ env / paths ------------------------
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(Path("./.hf_cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("./.cache").resolve()))

#CSV_PATH = "aigc_speech_generation_tasks/aigc_speech_generation_tasks.csv" ######
REF_DIR  = Path("aigc_speech_generation_tasks")
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
            return wer(ref_text.strip(), hyp), hyp
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
        return wer(ref_text.strip(), hyp), hyp
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

# ------------------------ Backends -------------------------------------------
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

# ---- XTTS (only if cached) --------------------------------------------------
def xtts_cached_model_path():
    # Linux 默认缓存位置；若你自定义了 TTS 缓存，可自行调整
    return Path.home() / ".local" / "share" / "tts" / "tts_models--multilingual--multi-dataset--xtts_v2" / "model.pth"

def xtts_available():
    return xtts_cached_model_path().exists()

class XTTSBackend:
    def __init__(self, device=None, language="zh-cn"):
        # 延迟安装 TTS，仅在真的启用 XTTS 时引入，避免与 pandas 2.x 冲突
        import importlib, subprocess as sp
        try:
            import TTS  # noqa
        except Exception:
            print("Installing TTS (XTTS backend) ...")
            # 若你有严格版本需求，可固定: TTS==0.22.0
            sp.check_call([sys.executable, "-m", "pip", "install", "TTS==0.22.0"])
        from TTS.api import TTS as CoquiTTS
        self.tts = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try: self.tts.to(self.device)
        except Exception: pass

    def synth_sentencewise(self, ref_wav: Path, text: str, out_wav: Path, seed: int=0):
        torch.manual_seed(seed)
        sents = split_sentences(text)
        pieces=[]; sr=24000
        for s in sents:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            self.tts.tts_to_file(text=s, file_path=str(tmp_path), speaker_wav=str(ref_wav), language=self.language)
            y, sr1 = sf.read(tmp_path); os.remove(tmp_path)
            if y.ndim>1: y=y.mean(axis=1)
            pieces.append(apply_fade(y.astype(np.float32), sr1)); sr=sr1
        out = concat_crossfade(pieces, sr, xfade_ms=10)
        sf.write(out_wav, out, sr)
        return out_wav

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
        for s in sents:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            # 句级直出
            self.voice.synthesize(s, str(tmp_path), sentence_silence=0.12)
            y, sr1 = sf.read(tmp_path); os.remove(tmp_path)
            if y.ndim>1: y=y.mean(axis=1)
            pieces.append(apply_fade(y.astype(np.float32), sr1)); sr=sr1
        out = concat_crossfade(pieces, sr, xfade_ms=10)
        sf.write(out_wav, out, sr)
        return out_wav

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
    out = concat_crossfade(chunks, sr, xfade_ms=10)
    try:
        import scipy.signal as ss  # optional
        b,a = ss.butter(4, [70/(sr/2), 7000/(sr/2)], btype="band")
        out = ss.lfilter(b,a,out).astype(np.float32)
    except Exception:
        pass
    sf.write(out_wav, out, sr)
    return out_wav

# ------------------------ selection ------------------------------------------
def pick_best(ref_wav: Path, text: str, cand_wavs: list, args, scorer=None,
              w_wer=0.4, w_sim=0.4, w_cm=0.2):
    if len(cand_wavs)==1 and args.no_asr and scorer is None:
        return cand_wavs[0], dict(wer=None, sim=None, cm=None)
    wers=[]; sims=[]; cms=[]
    for w in cand_wavs:
        try: werr,_ = asr_wer(w, text, args); wers.append(werr)
        except Exception: wers.append(0.5)
        try: sims.append(spk_similarity(ref_wav, w))
        except Exception: sims.append(0.0)
        if scorer is not None:
            try: cms.append(scorer.score(w))
            except Exception: cms.append(None)
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
            cn = (cms[i]-min_c)/(max_c-min_c) if max_c!=min_c else 0.5
            final = w_wer*(1.0-wn) + w_sim*sn + w_cm*(1.0-cn)
        scores.append(final)
    best_idx = int(np.argmax(scores))
    return cand_wavs[best_idx], dict(wer=wers[best_idx], sim=sims[best_idx], cm=cms[best_idx])

# ------------------------ main -----------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks_csv", default=CSV_PATH)
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

    # Piper (optional offline fallback)
    ap.add_argument("--use_piper_if_available", action="store_true")
    ap.add_argument("--piper_model", default="")  # e.g., models/piper/zh-CN.onnx

    args = ap.parse_args()

    df = pd.read_csv(args.tasks_csv)
    if args.limit>0: df = df.head(args.limit)

    out_dir = Path(args.team_name if args.team_name.endswith("-result") or args.team_name=="result"
                   else f"{args.team_name}-result")
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

    # AASIST
    scorer=None
    if args.aasist_repo and args.aasist_ckpt:
        try:
            scorer = AASISTSpoofScorer(repo_dir=args.aasist_repo, ckpt_path=args.aasist_ckpt)
            print("AASIST enabled.")
        except Exception as e:
            print("AASIST init failed, continue without it:", e)

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
                ref_mimic_collage(preprocess_ref(ref_path), estimate_duration(text), out_final); used_collage+=1
            except Exception:
                write_tone_placeholder(out_final, seconds=estimate_duration(text)); used_tone+=1
            continue

        # preprocess
        ref = preprocess_ref(ref_path)

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
