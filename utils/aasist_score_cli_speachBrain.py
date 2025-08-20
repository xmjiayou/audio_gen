#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AASIST score CLI (适配 TakHemlata/SSL_Anti-spoofing，并支持 SpeechBrain 回退/强制)
用法示例：
  # 1) 强制走 SpeechBrain（最稳，不依赖 fairseq/hydra）
  python utils/aasist_score_cli.py --backend speechbrain --wav some.wav

  # 2) 先尝试 repo（import model），失败再回退 SpeechBrain
  python utils/aasist_score_cli.py --repo /path/to/SSL_Anti-spoofing \
         --ckpt /path/to/best_model.pth --wav some.wav

无论成功失败，最后一行都会输出 JSON，且退出码为 0：
  成功：{"cm_score": <0..1>, "backend": "repo"|"speechbrain"}
  失败兜底：{"cm_score": 0.5, "backend": "fallback", "err": "..."}
"""
import argparse, json, sys, os

def _safe_print_json(obj):
    print(json.dumps(obj, ensure_ascii=False))
def _speechbrain_prob(wav_path: str, device: str) -> float:
    """
    使用 SpeechBrain 预训练接口进行反欺骗打分。
    兼容 source:
      - "speechbrain/antispoofing"
      - "speechbrain/antispoofing-AASIST"
    返回 spoof 概率（0..1）。
    """
    import numpy as np
    from speechbrain.pretrained import CMClassifier  # 预训练反欺骗分类器

    last_err = None
    for src in ["speechbrain/antispoofing", "speechbrain/antispoofing-AASIST"]:
        try:
            cm = CMClassifier.from_hparams(source=src, run_opts={"device": device})
            # classify_file 返回字典/张量，因模型不同而异，这里做兼容
            out = cm.classify_file(wav_path)

            # 规范化为 numpy 向量
            if isinstance(out, dict):
                score = out.get("score", None)
                if score is None:
                    # 有些模型返回 "prediction" / "scores"
                    score = out.get("scores", out.get("prediction", None))
                if score is None:
                    raise RuntimeError(f"Unexpected SB output keys: {list(out.keys())}")
                try:
                    import torch
                    if hasattr(score, "detach"):
                        score = score.detach().cpu().numpy()
                except Exception:
                    pass
                arr = np.array(score).reshape(-1)
            else:
                arr = np.array(out).reshape(-1)

            # 2-logit 情况：softmax，取第0维为 spoof 概率；单值：sigmoid 后取 1- bonafide
            if arr.size >= 2:
                ex = np.exp(arr - np.max(arr))
                prob = ex / (ex.sum() + 1e-9)
                prob_spoof = float(prob[0])  # 约定 index0 是 spoof
            else:
                s = float(1.0 / (1.0 + np.exp(-arr[0])))  # bonafide 概率
                prob_spoof = float(1.0 - s)
            return prob_spoof
        except Exception as e:
            last_err = e

    raise RuntimeError(f"speechbrain predict failed: {last_err}")

# def _speechbrain_prob(wav_path: str, device: str) -> float:
#     import numpy as np
#     from speechbrain.inference.AntiSpoof import AntiSpoof
#     # 尝试两个预训练源名
#     last_err = None
#     for src in ["speechbrain/antispoofing", "speechbrain/antispoofing-AASIST"]:
#         try:
#             model = AntiSpoof.from_hparams(source=src, run_opts={"device": device})
#             out = model.predict_file(wav_path)
#             # 兼容字典或张量/数组
#             if isinstance(out, dict):
#                 score = out.get("score", None)
#                 if score is None:
#                     raise RuntimeError("No 'score' in speechbrain output")
#                 try:
#                     import torch
#                     if hasattr(score, "detach"):
#                         score = score.detach().cpu().numpy()
#                 except Exception:
#                     pass
#                 arr = np.array(score).reshape(-1)
#             else:
#                 arr = np.array(out).reshape(-1)

#             if arr.size >= 2:
#                 # logits -> softmax，取第0维视作 spoof 概率
#                 ex = np.exp(arr - np.max(arr))
#                 prob = ex / (ex.sum() + 1e-9)
#                 prob_spoof = float(prob[0])
#             else:
#                 # 单通道分数 -> sigmoid 当作 bonafide 概率，再取 1 - p 作为 spoof
#                 s = float(1.0 / (1.0 + np.exp(-arr[0])))
#                 prob_spoof = float(1.0 - s)
#             return prob_spoof
#         except Exception as e:
#             last_err = e
#     raise RuntimeError(f"speechbrain predict failed: {last_err}")

def _repo_prob(repo_root: str, ckpt_path: str, wav_path: str, device: str) -> float:
    import importlib, numpy as np
    import soundfile as sf, librosa, torch, torch.nn.functional as F
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    ssl_model = importlib.import_module("model")  # TakHemlata/SSL_Anti-spoofing 根目录的 model.py

    net = ssl_model.Model(device)
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state, strict=False)
    net.eval()

    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000
    x = torch.from_numpy(y).float()[None, None, :].to(device)

    with torch.no_grad():
        out = net(x)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        if out.ndim == 1:
            out = out.unsqueeze(0)
        if out.shape[-1] == 2:
            prob_spoof = F.softmax(out, dim=-1)[0, -1].item()
        else:
            prob_spoof = torch.sigmoid(out[0, 0]).item()
    return float(prob_spoof)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto", "repo", "speechbrain"], default="auto",
                    help="选择打分后端：auto=先repo再回退；repo=只用repo；speechbrain=只用SpeechBrain")
    ap.add_argument("--repo", help="SSL_Anti-spoofing 仓库根目录（backend=repo/auto 时需要）")
    ap.add_argument("--ckpt", help="AASIST 模型权重 .pth（backend=repo/auto 时建议）")
    ap.add_argument("--wav", required=True, help="待评估 wav")
    args = ap.parse_args()

    # 选择设备
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    try:
        if args.backend == "speechbrain":
            score = _speechbrain_prob(args.wav, device)
            _safe_print_json({"cm_score": float(score), "backend": "speechbrain"})
            return 0

        if args.backend == "repo":
            if not args.repo:
                raise ValueError("--repo is required when backend=repo")
            score = _repo_prob(args.repo, args.ckpt, args.wav, device)
            _safe_print_json({"cm_score": float(score), "backend": "repo"})
            return 0

        # auto：先 repo，失败就 speechbrain
        try:
            if not args.repo:
                raise RuntimeError("skip repo: no --repo provided")
            score = _repo_prob(args.repo, args.ckpt, args.wav, device)
            _safe_print_json({"cm_score": float(score), "backend": "repo"})
            return 0
        except Exception as e_repo:
            try:
                score = _speechbrain_prob(args.wav, device)
                _safe_print_json({"cm_score": float(score), "backend": "speechbrain",
                                  "warn": f"repo_failed: {e_repo.__class__.__name__}"})
                return 0
            except Exception as e_sb:
                _safe_print_json({"cm_score": 0.5, "backend": "fallback",
                                  "err": f"repo_failed={repr(e_repo)}; speechbrain_failed={repr(e_sb)}"})
                return 0

    except Exception as e:
        _safe_print_json({"cm_score": 0.5, "backend": "fallback", "err": repr(e)})
        return 0

if __name__ == "__main__":
    try:
        code = main()
    except Exception as e:
        _safe_print_json({"cm_score": 0.5, "backend": "fallback", "err": repr(e)})
        code = 0
    sys.exit(code)
