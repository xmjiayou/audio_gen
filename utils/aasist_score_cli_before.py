#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调用方式：
  /path/to/aasist_env/bin/python utils/aasist_score_cli.py \
      --repo /path/to/SSL_Anti-spoofing \
      --ckpt /path/to/best_SSL_model_LA.pth \
      --wav some.wav
stdout 最后一行输出 JSON：{"cm_score": 0.123}
"""
import argparse, json, sys, os
import numpy as np
import soundfile as sf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wav", required=True)
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.repo))
    import torch
    import librosa
    # ↓↓↓ AASIST 官方/复现代码可能是这些命名（按你仓库实际改）
    import model as ssl_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ssl_model.Model(device)
    state = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(state, strict=False)
    net.eval()

    # 读音频并转 16k 单声道
    y, sr = sf.read(args.wav)
    if y.ndim > 1: y = y.mean(axis=1)
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

        # 若是二分类 logits：[bonafide, spoof]，取 spoof 概率为 cm_score
        if out.shape[-1] == 2:
            import torch.nn.functional as F
            prob_spoof = F.softmax(out, dim=-1)[0, -1].item()
            cm = float(prob_spoof)
        else:
            # 有些实现是单通道 sigmoid
            cm = float(torch.sigmoid(out[0, 0]).item())

    print(json.dumps({"cm_score": cm}, ensure_ascii=False))

if __name__ == "__main__":
    main()
