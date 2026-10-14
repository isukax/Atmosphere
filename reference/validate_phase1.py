"""Phase 1 の数値検証（docs/design.md §8.2 V5 / V6）。

アプリをバッチモードで起動して GPU の結果をダンプさせ、Python の参照実装と比較する。

    reference/.venv/Scripts/python.exe reference/validate_phase1.py

  V5: GT7 Tone Mapping（シェーダー） vs gt7_tone_mapping.py（元 C++ と照合済み）
      格子点で知覚色差 ΔE_ITP（ITU-R BT.2124）< 0.5。チャンネルごとの絶対誤差は参考値として記録する。
      ※ チャンネル単位の誤差を基準にしない理由: 高彩度色では ICtCp → RGB の逆変換で
        ほぼ 0 のチャンネルが桁落ちし、float32 と float64 の Python 参照どうしでも
        PQ 10bit コードで最大 70 コード（0.01 nit 程度）ずれる。表示上は無意味な差なので、
        画素全体の知覚色差で判定する（float32 参照 vs float64 参照の ΔE_ITP は最大 0.07）。
  V6: PQ エンコード（シェーダー → R10G10B10A2） vs 倍精度の ST 2084
      10bit コードの差 ≤ 1 LSB
  補助: 色空間行列（C++ の定数） vs colour-science
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exrio import read_exr  # noqa: E402
from gt7_tone_mapping import tone_mapper  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXE = ROOT / "build" / "vs2026-x64" / "Debug" / "atmosphere.exe"

GT7_DELTA_E_ITP_TOLERANCE = 0.5
PQ_CODE_TOLERANCE = 1
MATRIX_TOLERANCE = 1e-5


def run_app(exe: Path, test: str, out_dir: Path) -> None:
    result = subprocess.run([str(exe), "--test", test, "--out", str(out_dir)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{exe.name} --test {test} failed:\n{result.stdout}\n{result.stderr}")


def pq_encode(nits: np.ndarray) -> np.ndarray:
    """ST 2084 の逆 EOTF（倍精度）。入力 nit、出力 [0,1]。"""

    m1, m2 = 0.1593017578125, 78.84375
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    y = np.clip(nits.astype(np.float64) / 10000.0, 0.0, 1.0)
    ym = np.power(y, m1)
    return np.power((c1 + c2 * ym) / (1.0 + c3 * ym), m2)


def rec2020_nits_to_ictcp(nits: np.ndarray) -> np.ndarray:
    """linear Rec.2020 [nit] → ICtCp（倍精度、BT.2100 の定義）。"""

    nits = np.maximum(nits.astype(np.float64), 0.0)
    to_lms = np.array([[1688, 2146, 262], [683, 2951, 462], [99, 309, 3688]], dtype=np.float64) / 4096.0
    lms_pq = pq_encode(nits @ to_lms.T)
    to_ictcp = np.array([[2048, 2048, 0], [6610, -13613, 7003], [17933, -17390, -543]], dtype=np.float64) / 4096.0
    return lms_pq @ to_ictcp.T


def delta_e_itp(a_nits: np.ndarray, b_nits: np.ndarray) -> np.ndarray:
    """ITU-R BT.2124 の ΔE_ITP。1 がおおよそ弁別閾（JND）。T = 0.5·Ct。"""

    d = rec2020_nits_to_ictcp(a_nits) - rec2020_nits_to_ictcp(b_nits)
    return 720.0 * np.sqrt(d[..., 0] ** 2 + (0.5 * d[..., 1]) ** 2 + d[..., 2] ** 2)


def check_gt7(out_dir: Path) -> list[dict]:
    inputs, _ = read_exr(out_dir / "gt7_input.exr")
    rgb_in = inputs[..., :3]
    results = []
    for mode in ("sdr", "hdr450", "hdr1000", "hdr2000"):
        gpu, meta = read_exr(out_dir / f"gt7_{mode}.exr")
        gpu = gpu[..., :3].astype(np.float64)
        ref = tone_mapper(mode).apply(rgb_in).astype(np.float64)
        # 出力単位 → 表示輝度 [nit]: SDR は 1.0 = 250 nit（GT7 の SDR 白）、HDR は fb ×100。
        to_nits = 250.0 if mode == "sdr" else 100.0
        de = delta_e_itp(gpu * to_nits, ref * to_nits)
        worst = np.unravel_index(np.argmax(de), de.shape)
        results.append(
            {
                "check": f"V5 GT7 {mode}",
                "units": meta.get("units"),
                "max_delta_e_itp": float(de.max()),
                "mean_delta_e_itp": float(de.mean()),
                "max_abs_error_info": float(np.abs(gpu - ref).max()),
                "worst_input": rgb_in[worst].tolist(),
                "worst_gpu": gpu[worst].tolist(),
                "worst_ref": ref[worst].tolist(),
                "tolerance": GT7_DELTA_E_ITP_TOLERANCE,
                "pass": bool(de.max() < GT7_DELTA_E_ITP_TOLERANCE),
            }
        )
    return results


def check_pq(out_dir: Path) -> list[dict]:
    inputs, _ = read_exr(out_dir / "pq_input.exr")
    codes, _ = read_exr(out_dir / "pq_codes.exr")
    # 入力は fb（1.0 = 100 nit）。トーンマップなしで ×100 → nit → PQ。
    nits = inputs[0, :, :3].astype(np.float64) * 100.0
    ref_codes = np.round(pq_encode(nits) * 1023.0)
    diff = np.abs(codes[0, :, :3].astype(np.float64) - ref_codes)
    return [
        {
            "check": "V6 PQ 10bit",
            "max_code_diff": float(diff.max()),
            "count_nonzero_diff": int(np.count_nonzero(diff)),
            "samples": int(diff.size),
            "tolerance": PQ_CODE_TOLERANCE,
            "pass": bool(diff.max() <= PQ_CODE_TOLERANCE),
        }
    ]


def check_matrices(out_dir: Path) -> list[dict]:
    import colour

    data = json.loads((out_dir / "matrices.json").read_text(encoding="utf-8"))
    bt709 = colour.RGB_COLOURSPACES["ITU-R BT.709"]
    bt2020 = colour.RGB_COLOURSPACES["ITU-R BT.2020"]
    expected = {
        "rec709_to_rec2020": colour.matrix_RGB_to_RGB(bt709, bt2020),
        "rec2020_to_rec709": colour.matrix_RGB_to_RGB(bt2020, bt709),
    }
    results = []
    for name, ref in expected.items():
        diff = float(np.abs(np.asarray(data[name]) - ref).max())
        results.append({"check": f"matrix {name}", "max_abs_error": diff, "tolerance": MATRIX_TOLERANCE,
                        "pass": diff < MATRIX_TOLERANCE})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--out", type=Path, default=ROOT / "dumps" / "phase1")
    args = parser.parse_args()

    run_app(args.exe, "gt7", args.out / "gt7")
    run_app(args.exe, "pq", args.out / "pq")

    results = check_gt7(args.out / "gt7") + check_pq(args.out / "pq") + check_matrices(args.out / "gt7")
    (args.out / "report.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for r in results:
        metric = r.get("max_delta_e_itp", r.get("max_abs_error", r.get("max_code_diff")))
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['check']:<28} max={metric:.3e}  tol={r['tolerance']}")
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
