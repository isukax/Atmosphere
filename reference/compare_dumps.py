"""2 つのダンプのディレクトリの EXR を画素で比べる（回帰の確認。design.md §5.9 の「回帰」）。

    reference/.venv/Scripts/python.exe reference/compare_dumps.py dumps/baseline_4_7 dumps/regress_4_8 [--tol 1e-6]

同じ相対パスの *.exr をすべて読み、最大の相対差（|a − b| / max(|a|, 1e-6)）と最大の絶対差を出す。
サイドカー（*.meta.json）は比べない（パラメータの項目が増えても画素が同じなら回帰なし）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exrio import read_exr  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--tol", type=float, default=1e-6, help="最大の相対差の許容値")
    args = parser.parse_args()
    files = sorted(p.relative_to(args.a) for p in args.a.rglob("*.exr"))
    worst = 0.0
    failed = []
    missing = []
    for rel in files:
        pb = args.b / rel
        if not pb.exists():
            missing.append(rel)
            continue
        a, _ = read_exr(args.a / rel)
        b, _ = read_exr(pb)
        if a.shape != b.shape:
            failed.append((rel, "shape", a.shape, b.shape))
            continue
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        diff = np.abs(a - b)
        rel_diff = float(np.max(diff / np.maximum(np.abs(a), 1e-6))) if a.size else 0.0
        worst = max(worst, rel_diff)
        if rel_diff > args.tol:
            failed.append((rel, rel_diff, float(diff.max())))
    print(f"{len(files)} files, worst relative difference {worst:.3e}")
    for f in failed:
        print("  DIFF", *f)
    for m in missing:
        print("  MISSING", m)
    return 0 if not failed and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
