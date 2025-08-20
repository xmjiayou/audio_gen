#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
适配 TakHemlata/SSL_Anti-spoofing 的 AASIST 打分 CLI
- 只依赖仓库根目录的 `model.py`（即 import model）
- 失败时也会输出 JSON 并以 0 退出，避免主进程报 CalledProcessError

用法：
  /path/to/aasist_env/bin/python utils/aasist_score_cli.py \
      --repo /path/to/SSL_Anti-spoofing \
      --ckpt /path/to/best_SSL_model_LA.pth \
      --wav  /path/to/example.wav

stdout 最后一行固定输出：
  {"cm_score": <0..1>, "backend": "repo"}    # 成功
或
  {"cm_score": 0.5, "backend": "fallback", "err": "..."}  # 失败兜底
"""
import argparse, json, sys, os

def _safe_print_json(obj):
    print(json.dumps(obj, ensure_ascii=False))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="本地克隆的 SSL_Anti-spoofing 仓库根目录")
    ap.add_argument("--ckpt", required=True, help="AASIST 模型权重 .pth 路径")
    ap.add_argument("--wav",  required=True, help="待评估的 wav 路径（任意采样率，自动重采样到16k）")
    args = ap.parse_args()

    try:
        # 1) 把 repo 根目录加到 sys.path，专门适配 TakHemlata/SSL_Anti-spoofing
        repo_root = os.path.abspath(args.repo)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        # 2) 仅尝试 import model（该仓库的 model.py 在根目录）
        import importlib
        ssl_model = importlib.import_module("model")

        # 3) 设备
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

        # 4) 构建与加载权重
        import torch
        net = ssl_model.Model(device)
        state = torch.load(args.ckpt, map_location=device)
        net.load_state_dict(state, strict=False)
        net.eval()

        # 5) 读 wav → 单声道 → 16k → Tensor[N=1,C=1,T]
        import numpy as np
        import soundfile as sf
        y, sr = sf.read(args.wav)
        if y is None:
            raise RuntimeError("failed to read wav")
        if hasattr(y, "ndim") and y.ndim > 1:
            import numpy as _np
            y = _np.mean(y, axis=1)
        if sr != 16000:
            import librosa
            y = librosa.resample(y.astype("float32"), orig_sr=sr, target_sr=16000)
            sr = 16000
        x = torch.from_numpy(y).float()[None, None, :].to(device)

        # 6) 前向 + 兼容不同输出形状
        with torch.no_grad():
            out = net(x)
            # 某些实现可能返回 tuple/list，取最后一个
            if isinstance(out, (list, tuple)):
                out = out[-1]
            if out.ndim == 1:
                out = out.unsqueeze(0)

            if out.shape[-1] == 2:
                # 二分类 logits：[bonafide, spoof]，取 spoof 概率
                import torch.nn.functional as F
                prob_spoof = F.softmax(out, dim=-1)[0, -1].item()
                cm = float(prob_spoof)
            else:
                # 单通道分数 → sigmoid
                cm = float(torch.sigmoid(out[0, 0]).item())

        _safe_print_json({"cm_score": cm, "backend": "repo"})
        return 0

    except Exception as e:
        # 失败兜底：不抛异常，避免主程序崩。输出中性分与错误信息用于日志排查
        _safe_print_json({"cm_score": 0.5, "backend": "fallback", "err": repr(e)})
        return 0

if __name__ == "__main__":
    try:
        code = main()
    except Exception as e:
        _safe_print_json({"cm_score": 0.5, "backend": "fallback", "err": repr(e)})
        code = 0
    sys.exit(code)
