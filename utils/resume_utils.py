# -*- coding: utf-8 -*-
"""
resume_utils.py
- 扫描 result 目录中已产出的“正式” wav（不含 'tmp'），支持基本有效性检查。
- 返回已完成 utt 的集合，供主脚本跳过已完成样本，实现断点续跑。
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Dict, Set, Tuple

# 可选：用 soundfile 读时长更稳；没有就退化成尺寸阈值检查
try:
    import soundfile as sf
    _HAS_SF = True
except Exception:
    _HAS_SF = False

_WAV_RE = re.compile(r"^(\d+)\.wav$", re.IGNORECASE)

def _is_tmp_name(name: str) -> bool:
    return "tmp" in name.lower()

def _looks_valid_wav(path: Path, min_bytes: int = 4096, min_sec: float = 0.1) -> bool:
    """基础有效性：不是 0 字节、足够大；若装了 soundfile，再检查能否读且时长>阈值。"""
    try:
        if not path.exists() or not path.is_file():
            return False
        if path.stat().st_size < min_bytes:
            return False
        if _HAS_SF:
            with sf.SoundFile(str(path)) as f:
                dur = len(f) / float(f.samplerate or 1)
                if dur < min_sec:
                    return False
        return True
    except Exception:
        return False

def list_official_wavs(result_dir: str | Path) -> Dict[int, str]:
    """
    扫描 result_dir，找形如 '123.wav' 的正式文件（排除文件名含 tmp），并做基本有效性检查。
    返回 {utt:int -> filepath:str}
    """
    result_dir = Path(result_dir)
    done: Dict[int, str] = {}
    if not result_dir.exists():
        return done

    for p in result_dir.iterdir():
        name = p.name
        if not p.is_file():
            continue
        if _is_tmp_name(name):
            continue
        m = _WAV_RE.match(name)
        if not m:
            continue
        utt = int(m.group(1))
        if _looks_valid_wav(p):
            done[utt] = str(p)
    return done

def summarize_progress(result_dir: str | Path, total: int = 200) -> Tuple[int, int]:
    """
    返回 (已完成个数, 剩余个数)
    """
    done = list_official_wavs(result_dir)
    n_done = len(done)
    return n_done, max(0, total - n_done)

def build_missing_set(total_utt: int, done: Set[int]) -> Set[int]:
    """
    给定总题数 total_utt（例如 200），以及已完成集合 done，返回缺失集合。
    """
    return set(range(1, total_utt + 1)) - set(done)
