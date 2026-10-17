"""確認画像の書き出し（外部ライブラリなしの 8 bit RGB PNG）と、簡易トーンマップ（GT7 ではない）。"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def tonemap_srgb(linear: np.ndarray) -> np.ndarray:
    """x/(1+x) で圧縮して sRGB にする（確認画像用。露出は呼び出し側で掛ける）。"""

    x = np.maximum(np.asarray(linear, dtype=np.float64), 0.0)
    x = x / (1.0 + x)
    return np.clip(np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055), 0.0, 1.0)


def write_png(path: Path, rgb01: np.ndarray) -> None:
    img = (np.clip(rgb01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
