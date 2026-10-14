"""EXR + サイドカー JSON の読み込み（アプリの src/io/ExrWriter.cpp と対になる）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import OpenEXR


def read_exr(path: Path) -> tuple[np.ndarray, dict]:
    """float32 RGBA 画像（形状 (h, w, 4)）とサイドカーのメタデータを返す。"""

    with OpenEXR.File(str(path)) as exr:
        channels = exr.channels()
        if "RGBA" in channels:
            rgba = np.asarray(channels["RGBA"].pixels, dtype=np.float32)
        else:
            rgba = np.stack([np.asarray(channels[c].pixels, dtype=np.float32) for c in "RGBA"], axis=-1)
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return rgba, meta
