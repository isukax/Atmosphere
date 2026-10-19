"""Phase 4.10 の検証（docs/design.md §5.10、UI から AOD を変える）。

    reference/.venv/Scripts/python.exe reference/validate_phase4_10.py [--skip-run] [--only ...]

アプリの aerosol_presets の、AOD を上書きした組（Phase48Tests.cpp の kCombos の aod550 > 0）を確かめる。
上書きしない組は、Phase 4.9 のダンプとの画素の回帰（compare_dumps.py）で確かめるので、ここでは扱わない。
  AOD: サイドカーの σs・σa・鉛直分布から求めた AOD が目標の値と一致する（Hillaire2020 は G = 550 nm なので AOD550、
       分光フィットのセットはプリセットの aod_rgb × 倍率と比べる）。σs・σa はプリセットの値 × 倍率（全成分・RGB）で、
       鉛直分布・位相関数・g はプリセットのまま。相対差 < 1e-5。
  V3i / V3j: validate_phase4_8.py と同じ関数・同じ基準。基準は AOD 0.1 までで決めたもので、0.3 は記録として扱う
       （合否は付けるが、外れたら品質の設定の問題として記録する。design.md §5.10）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aerosol_models as am  # noqa: E402
import validate_phase4_8 as v48  # noqa: E402
from exrio import read_exr  # noqa: E402
from validate_phase1 import ROOT, run_app  # noqa: E402

# Phase48Tests.cpp の kCombos の AOD を上書きした組
AOD_COMBOS = (("Hillaire2020", "C+A+B", 0.02), ("Hillaire2020", "C+A+B", 0.3), ("Hillaire2020", "C+A+B+D", 0.02),
              ("Hillaire2020", "C+A+B+D", 0.3), ("SpectralFit709", "C+A+B+D", 0.3))
AOD_TOLERANCE = 1e-5
VALIDATED_AOD = 0.1  # V3i・V3j の基準を決めた AOD（これより濃い組は記録）


def sidecar_components(a: dict) -> list[dict]:
    """サイドカーの "atmosphere" から、Mie の成分ごとの σs・σa・鉛直分布・g"""

    comps = [{"scattering": np.asarray(a["mie_scattering_per_km"], dtype=np.float64),
              "absorption": np.asarray(a["mie_absorption_per_km"], dtype=np.float64),
              "profile": am.Profile(float(a["mie_boundary_layer_top_km"]), float(a["mie_scale_height_km"]),
                                    float(a["mie_free_troposphere_density"]),
                                    float(a["mie_free_troposphere_scale_height_km"])),
              "g": float(a["mie_phase_g"]), "phase_model": a["mie_phase_model"]}]
    if int(a.get("mie_component_count", 1)) == 2:
        m = a["mie1"]
        comps.append({"scattering": np.asarray(m["scattering_per_km"], dtype=np.float64),
                      "absorption": np.asarray(m["absorption_per_km"], dtype=np.float64),
                      "profile": am.Profile(float(m["boundary_layer_top_km"]), float(m["scale_height_km"]),
                                            float(m["free_troposphere_density"]),
                                            float(m["free_troposphere_scale_height_km"])),
                      "g": float(m["phase_g"]), "phase_model": m["phase_model"]})
    return comps


def check_aod(out: Path, combo, coeff: dict) -> dict:
    set_name, preset_name, target = combo
    _, meta = read_exr(out / "transmittance" / f"{v48.stem(combo)}.exr")
    a = meta["atmosphere"]
    s = next(x for x in coeff["sets"] if x["name"] == set_name)
    preset = next(x for x in s["aerosol_presets"] if x["name"] == preset_name)
    k = target / preset["aod550"]
    comps = sidecar_components(a)
    if len(comps) != len(preset["components"]):
        return {"check": f"AOD override ({v48.stem(combo)})", "info": "component count differs", "pass": False}

    f32 = lambda x: np.asarray(x, dtype=np.float32).astype(np.float64)  # noqa: E731  JSON の値は float32 で読まれる
    sigma_err = 0.0
    shape_same = True
    for c, p in zip(comps, preset["components"]):
        for key, json_key in (("scattering", "scattering_per_km"), ("absorption", "absorption_per_km")):
            expected = f32(p[json_key]) * k
            sigma_err = max(sigma_err, float(np.max(np.abs(c[key] - expected) / expected)))
        shape_same &= (c["profile"].top_km == f32(p["boundary_layer_top_km"])
                       and c["profile"].scale_height_km == f32(p["scale_height_km"])
                       and c["profile"].free_trop_density == f32(p["free_troposphere_density"])
                       and c["profile"].free_trop_scale_height_km == f32(p["free_troposphere_scale_height_km"])
                       and c["phase_model"] == p["phase"]["model"])
        ph = p["phase"]
        expected_g = ph["g"] if ph["model"] == "cornette_shanks" else ph["asymmetry_g"][am.G_CHANNEL]
        shape_same &= c["g"] == float(np.float32(expected_g))

    aod_rgb = sum(am.column_km(c["profile"]) * (c["scattering"] + c["absorption"]) for c in comps)
    if set_name == "Hillaire2020":  # G = 550 nm
        aod_err = abs(aod_rgb[am.G_CHANNEL] - target) / target
    else:
        expected_rgb = np.asarray(preset["aod_rgb"]) * k
        aod_err = float(np.max(np.abs(aod_rgb - expected_rgb) / expected_rgb))
    recorded_err = abs(float(a["aerosol_aod550"]) - target) / target
    flags_ok = (a["aerosol_aod550_overridden"] is True and a["aerosol_preset_modified"] is True
                and a["aerosol_preset_name"] == preset_name and a["aerosol_preset"] == preset_name + " (modified)")
    worst = max(sigma_err, aod_err, recorded_err)
    return {"check": f"AOD override ({v48.stem(combo)})", "aod_rel_error": float(aod_err),
            "sigma_rel_error": sigma_err, "recorded_aod550_rel_error": recorded_err,
            "aod_rgb": aod_rgb.tolist(), "scale": k, "profile_and_phase_unchanged": bool(shape_same),
            "sidecar_flags_ok": bool(flags_ok), "tolerance": AOD_TOLERANCE,
            "pass": bool(worst < AOD_TOLERANCE and shape_same and flags_ok)}


def mark_unvalidated(result: dict, combo) -> dict:
    """AOD 0.1 より濃い組は記録として扱う（基準は 0.1 までで決めたもの）"""

    if combo[2] > VALIDATED_AOD:
        result["criterion_met"] = result["pass"]
        result["pass"] = True
        result["info"] = f"recorded only (AOD {combo[2]:g} > {VALIDATED_AOD:g}, design.md §5.10)"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", type=Path, default=ROOT / "build" / "vs2026-x64" / "Release" / "atmosphere.exe")
    parser.add_argument("--out", type=Path, default=ROOT / "dumps" / "phase4_10")
    parser.add_argument("--skip-run", action="store_true", help="既存のダンプだけを比較する")
    parser.add_argument("--only", nargs="*", default=None, help="AOD V3i V3j のうち実行するもの")
    args = parser.parse_args()
    if not args.skip_run:
        run_app(args.exe, "aerosol_presets", args.out)
    coeff = json.loads((ROOT / "data" / "atmosphere_coefficients.json").read_text(encoding="utf-8"))
    run = (lambda k: args.only is None or k in args.only)  # noqa: E731
    results = []
    for combo in AOD_COMBOS:
        if run("AOD"):
            results.append(check_aod(args.out, combo, coeff))
        if run("V3i"):
            results.append(mark_unvalidated(v48.check_transmittance(args.out, combo), combo))
            results += [mark_unvalidated(r, combo) for r in v48.check_single(args.out, combo)]
        if run("V3j"):
            results.append(mark_unvalidated(v48.check_ms(args.out, combo), combo))
        print(f"  {v48.stem(combo)} done", flush=True)
    tag = "" if args.only is None else "_" + "_".join(args.only)
    (args.out / f"report{tag}.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8", newline="\n")
    for r in results:
        keys = [k for k in r if k not in ("check", "pass", "tolerance", "info")]
        brief = ", ".join(f"{k}={r[k]:.3g}" if isinstance(r[k], float) else f"{k}={r[k]}" for k in keys[:3])
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['check']:<60} {brief[:160]}")
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
