# aasist_wrapper.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional

class AASISTScorer:
    """
    统一封装的 AASIST 评分器：
    - score_bonafide(wav_path) -> float in [0,1] （越大越“真人”）
    - 如果依赖不可用或推理失败，返回 None（上层自动忽略此指标）
    """
    def __init__(self, device: str = "auto"):
        self.device = device
        self._impl = None
        self._init_impl()

    def _init_impl(self):
        """
        优先使用 SpeechBrain 的 AASIST 模型。
        如果你的项目里已有自定义 AASIST 推理，请在这里接你的实现：
           - 把 self._impl 设成一个可调用：impl(wav_np, sr) -> bonafide_prob[0..1]
        """
        try:
            import torch
            dev = "cuda" if (self.device == "auto" and torch.cuda.is_available()) else self.device
            # SpeechBrain AASIST（模型 id 可能有差异，做两次尝试）
            try:
                from speechbrain.inference import SpeakerRecognition  # 避免旧版本 import 报错
            except Exception:
                pass

            # 官方 AntiSpoof 推理接口（不同版本的 SpeechBrain 可能模块路径不同）
            try:
                from speechbrain.inference.AntiSpoof import AntiSpoof
                self._impl = AntiSpoof.from_hparams(
                    source="speechbrain/antispoofing",  # 若加载失败会在下一步 fallback
                    run_opts={"device": dev}
                )
                self._mode = "sb_antispoof_generic"
                return
            except Exception:
                pass

            try:
                # 一些版本的仓库名/卡名不同，尝试 AASIST 关键词模型
                from speechbrain.inference.AntiSpoof import AntiSpoof
                self._impl = AntiSpoof.from_hparams(
                    source="speechbrain/antispoofing-AASIST",
                    run_opts={"device": dev}
                )
                self._mode = "sb_antispoof_aasist"
                return
            except Exception:
                pass

            # 都失败：标记为不可用
            self._impl = None
            self._mode = "none"
        except Exception:
            self._impl = None
            self._mode = "none"

    def available(self) -> bool:
        return self._impl is not None

    def score_bonafide(self, wav_path: str | Path) -> Optional[float]:
        """
        返回 bonafide 概率（0~1）。失败时返回 None。
        """
        if not self.available():
            return None
        try:
            if hasattr(self._impl, "predict_file"):
                # SpeechBrain AntiSpoof 通用接口（大多数版本可用）
                out = self._impl.predict_file(str(wav_path))
                # 约定返回字典或张量，这里尽量兼容
                # 1) 如果是 dict：{"score": tensor([spoof, bona]) 或类似}
                if isinstance(out, dict):
                    # 常见返回：{"score": tensor([[score_spoof, score_bonafide]])}
                    score = out.get("score", None)
                    if score is None:
                        return None
                    import torch
                    if isinstance(score, torch.Tensor):
                        score = score.detach().cpu().numpy()
                    arr = np.array(score).reshape(-1)
                    # 尝试取 bonafide 维：大多数实现是第 1 维
                    if arr.size >= 2:
                        # softmax 后第二维近似 bonafide
                        ex = np.exp(arr - np.max(arr))
                        prob = ex / (ex.sum() + 1e-9)
                        return float(prob[-1])
                    # 若只有一个分数，做一个映射（越大越真）
                    # 这里 sigmoid 归一
                    return float(1.0 / (1.0 + np.exp(-arr[0])))
                else:
                    # 可能返回 logits 向量
                    arr = np.array(out).reshape(-1)
                    if arr.size >= 2:
                        ex = np.exp(arr - np.max(arr))
                        prob = ex / (ex.sum() + 1e-9)
                        return float(prob[-1])
                    return float(1.0 / (1.0 + np.exp(-arr[0])))

            # 兜底：直接失败
            return None
        except Exception:
            return None
