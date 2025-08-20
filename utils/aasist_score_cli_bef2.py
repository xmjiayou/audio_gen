#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调用方式（与你现有一致）：
  /path/to/aasist_env/bin/python utils/aasist_score_cli.py \
      --repo /path/to/SSL_Anti-spoofing \
      --ckpt /path/to/best_SSL_model_LA.pth \
      --wav some.wav
stdout 最后一行始终输出 JSON：{"cm_score": 0.123, ...}
- 失败也会输出 {"cm_score": 0.5, "err": "..."} 且退出码为 0，避免主进程炸。
"""
import argparse, json, sys, os
import importlib
import numpy as np

def _safe_print_json(obj):
    # 保证最后一行是 JSON
    text = json.dumps(obj, ensure_ascii=False)
    print(text)

def _load_repo_model(repo_root: str, ckpt_path: str, device: str):
    """
    最大兼容地加载开源库模型；找不到就抛异常让上层 fallback。
    尝试的候选模块路径：
      - model: Model
      - models.AASIST.model: Model
      - AASIST.model: Model
    """
    sys.path.insert(0, os.path.abspath(repo_root))
    last_err = None
    candidates = [
        ("model", "Model"),
        ("models.AASIST.model", "Model"),
        ("AASIST.model", "Model"),
    ]
    for mod_name, cls_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            Net = getattr(mod, cls_name)
            import torch
            net = Net(device)
            state = torch.load(ckpt_path, map_location=device)
            net.load_state_dict(state, strict=False)
            net.eval()
            return ("repo", net)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"repo import failed: {last_err}")

def _repo_forward_to_prob(model_obj, wav_path: str, device: str) -> float:
    """
    将 repo 模型前向输出转为 [0..1] spoof 概率（或 bonafide 概率），
    约定：若为二分类 logits，取 softmax 的最后一维；若单通道，取 sigmoid。
    """
    import soundfile as sf, librosa, torch, torch.nn.functional as F
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000
    x = torch.from_numpy(y).float()[None, None, :].to(device)
    with torch.no_grad():
        out = model_obj(x)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        if out.ndim == 1:
            out = out.unsqueeze(0)
        if out.shape[-1] == 2:
            prob_spoof = F.softmax(out, dim=-1)[0, -1].item()
            return float(prob_spoof)
        else:
            return float(torch.sigmoid(out[0, 0]).item())

def _speechbrain_prob(wav_path: str, device: str) -> float:
    """
    回退：使用 SpeechBrain 的 AntiSpoof 推理。
    优先尝试 'speechbrain/antispoofing'，失败则试 'speechbrain/antispoofing-AASIST'
    返回 bonafide 概率（或等价分数）；为与上游统一，这里转换成「spoof 概率」：
      如果拿到 logits，softmax 后取第 0 维视作 spoof 概率；若只有 1 维，用 sigmoid。
    """
    import numpy as np
    try:
        from speechbrain.inference.AntiSpoof import AntiSpoof
    except Exception as e:
        raise RuntimeError(f"speechbrain import failed: {e}")

    last_err = None
    for src in ["speechbrain/antispoofing", "speechbrain/antispoofing-AASIST"]:
        try:
            model = AntiSpoof.from_hparams(source=src, run_opts={"device": device})
            out = model.predict_file(wav_path)
            # 兼容多种返回
            if isinstance(out, dict):
                score = out.get("score", None)
                if score is None:
                    raise RuntimeError("No 'score' in speechbrain output")
                try:
                    import torch
                    if isinstance(score, torch.Tensor):
                        score = score.detach().cpu().numpy()
                except Exception:
                    pass
                arr = np.array(score).reshape(-1)
            else:
                arr = np.array(out).reshape(-1)

            if arr.size >= 2:
                ex = np.exp(arr - np.max(arr))
                prob = ex / (ex.sum() + 1e-9)
                prob_spoof = float(prob[0])  # 约定第0维为spoof
            else:
                # 单通道打分：sigmoid 后视作 “越大越真(bonafide)”
                # 这里返回 spoof 概率 = 1 - bonafide_prob
                s = float(1.0 / (1.0 + np.exp(-arr[0])))
                prob_spoof = float(1.0 - s)
            return prob_spoof
        except Exception as e:
            last_err = e
    raise RuntimeError(f"speechbrain predict failed: {last_err}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wav", required=True)
    args = ap.parse_args()

    # 选择设备
    device = "cpu"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception as e_run:
        print("cuda error:", e_run)
        pass

    try:
        # 先尝试按 repo+ckpt 的方式
        kind, net = _load_repo_model(args.repo, args.ckpt, device)
        cm_spoof_prob = _repo_forward_to_prob(net, args.wav, device)
        _safe_print_json({"cm_score": float(cm_spoof_prob), "backend": "repo"})
        return 0
    except Exception as e_repo:
        # 回退到 speechbrain
        print("Can not try IN AASIST-origin, aasist score is 0.5", e_repo)
        try:
            cm_spoof_prob = _speechbrain_prob(args.wav, device)
            _safe_print_json({
                "cm_score": float(cm_spoof_prob),
                "backend": "speechbrain",
                "warn": f"repo_failed: {e_repo.__class__.__name__}"
            })
            return 0
        except Exception as e_sb:
            # 最后兜底：输出中性分 + 错误信息，避免上游崩
            _safe_print_json({
                "cm_score": 0.5,  # 中性分
                "backend": "fallback",
                "err": f"repo_failed={repr(e_repo)}; speechbrain_failed={repr(e_sb)}"
            })
            print("Can not try IN AASIST-speechbrain, aasist score is 0.5", e_sb)
            return 0

if __name__ == "__main__":
    # 保证任何异常都不会导致非零退出，避免上游 CalledProcessError
    try:
        code = main()
    except Exception as e:
        _safe_print_json({"cm_score": 0.5, "backend": "fallback", "err": repr(e)})
        print("Can not try IN AASIST, aasist score is 0.5", e)###
        code = 0
    sys.exit(code)
