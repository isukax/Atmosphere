# コピー元: GraphicsTest/LUTGenGT7/gt7_tone_mapping.py（無変更）
# 元の C++: Polyphony Digital "GT7 Tone Mapping" サンプル (Copyright (c) 2025 Polyphony Digital Inc., MIT License)
# LUTGenGT7 の validate.py で元の C++ とランダム照合済み。本プロジェクトではシェーダー (V5) の正解値として使う。
"""NumPy port of Polyphony Digital's GT7 tone-mapping sample.

The public functions operate on arrays whose final dimension is RGB.  Computation
is intentionally performed in float32 so the results track the original C++
implementation, which uses ``float`` throughout.

Input and output are linear Rec.2020 frame-buffer values.  One frame-buffer unit
corresponds to 100 cd/m^2.  No output OETF is applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray
F32 = np.float32

REFERENCE_LUMINANCE = F32(100.0)
GRAN_TURISMO_SDR_PAPER_WHITE = F32(250.0)

# DaVinci Intermediate constants from Blackmagic Design's published specification.
DI_A = F32(0.0075)
DI_B = F32(7.0)
DI_C = F32(0.07329248)
DI_M = F32(10.44426855)
DI_LIN_CUT = F32(0.00262409)
DI_LOG_CUT = F32(0.02740668)


def _f32(value: object) -> Array:
    return np.asarray(value, dtype=np.float32)


def frame_buffer_to_physical(value: object) -> Array:
    return _f32(value) * REFERENCE_LUMINANCE


def physical_to_frame_buffer(value: object) -> Array:
    return _f32(value) / REFERENCE_LUMINANCE


def davinci_intermediate_encode(linear: object) -> Array:
    """Encode relative linear light using the DaVinci Intermediate transfer curve."""

    linear = _f32(linear)
    # Avoid evaluating log2 for values on the linear branch that are below -DI_A.
    result = np.empty_like(linear)
    linear_mask = linear <= DI_LIN_CUT
    result[linear_mask] = linear[linear_mask] * DI_M
    log_values = linear[~linear_mask]
    result[~linear_mask] = (np.log2(log_values + DI_A) + DI_B) * DI_C
    return result


def davinci_intermediate_decode(encoded: object) -> Array:
    """Decode DaVinci Intermediate values to relative linear light."""

    encoded = _f32(encoded)
    result = np.empty_like(encoded)
    linear_mask = encoded <= DI_LOG_CUT
    result[linear_mask] = encoded[linear_mask] / DI_M
    log_values = encoded[~linear_mask]
    result[~linear_mask] = np.exp2(log_values / DI_C - DI_B) - DI_A
    return result


def smooth_step(x: object, edge0: object, edge1: object) -> Array:
    x = _f32(x)
    edge0 = F32(edge0)
    edge1 = F32(edge1)
    t = (x - edge0) / (edge1 - edge0)
    polynomial = t * t * (F32(3.0) - F32(2.0) * t)
    return np.where(x < edge0, F32(0.0), np.where(x > edge1, F32(1.0), polynomial)).astype(
        np.float32
    )


def chroma_curve(x: object, a: object, b: object) -> Array:
    return F32(1.0) - smooth_step(x, a, b)


def eotf_st2084(encoded: object, exponent_scale_factor: object = 1.0) -> Array:
    """PQ EOTF returning GT frame-buffer units (1.0 == 100 cd/m^2)."""

    n = np.clip(_f32(encoded), F32(0.0), F32(1.0))
    m1 = F32(0.1593017578125)
    m2 = F32(78.84375) * F32(exponent_scale_factor)
    c1 = F32(0.8359375)
    c2 = F32(18.8515625)
    c3 = F32(18.6875)

    np_value = np.power(n, F32(1.0) / m2)
    luminance = np.maximum(np_value - c1, F32(0.0))
    luminance = luminance / (c2 - c3 * np_value)
    luminance = np.power(luminance, F32(1.0) / m1)
    # Preserve the original operation order: normalize to 10000 nits, then
    # physicalValueToFrameBufferValue divides by 100 nits per unit.
    physical = luminance * F32(10000.0)
    return physical / REFERENCE_LUMINANCE


def inverse_eotf_st2084(linear: object, exponent_scale_factor: object = 1.0) -> Array:
    """Inverse PQ EOTF accepting GT frame-buffer units."""

    value = _f32(linear)
    m1 = F32(0.1593017578125)
    m2 = F32(78.84375) * F32(exponent_scale_factor)
    c1 = F32(0.8359375)
    c2 = F32(18.8515625)
    c3 = F32(18.6875)

    # Preserve the original C++ rounding points; PQ contains subtractive terms
    # that amplify otherwise tiny differences.
    physical = value * REFERENCE_LUMINANCE
    normalized = physical / F32(10000.0)
    ym = np.power(normalized, m1)
    return np.exp2(m2 * (np.log2(c1 + c2 * ym) - np.log2(F32(1.0) + c3 * ym)))


def rgb_to_ictcp(rgb: object) -> Array:
    """Convert non-negative linear Rec.2020 RGB to ICtCp."""

    rgb = _f32(rgb)
    r, g, b = np.moveaxis(rgb, -1, 0)
    l = (r * F32(1688.0) + g * F32(2146.0) + b * F32(262.0)) / F32(4096.0)
    m = (r * F32(683.0) + g * F32(2951.0) + b * F32(462.0)) / F32(4096.0)
    s = (r * F32(99.0) + g * F32(309.0) + b * F32(3688.0)) / F32(4096.0)

    l_pq = inverse_eotf_st2084(l)
    m_pq = inverse_eotf_st2084(m)
    s_pq = inverse_eotf_st2084(s)

    intensity = (F32(2048.0) * l_pq + F32(2048.0) * m_pq) / F32(4096.0)
    ct = (
        F32(6610.0) * l_pq - F32(13613.0) * m_pq + F32(7003.0) * s_pq
    ) / F32(4096.0)
    cp = (
        F32(17933.0) * l_pq - F32(17390.0) * m_pq - F32(543.0) * s_pq
    ) / F32(4096.0)
    return np.stack((intensity, ct, cp), axis=-1).astype(np.float32)


def ictcp_to_rgb(ictcp: object) -> Array:
    """Convert ICtCp to non-negative linear Rec.2020 RGB."""

    ictcp = _f32(ictcp)
    intensity, ct, cp = np.moveaxis(ictcp, -1, 0)
    l = intensity + F32(0.00860904) * ct + F32(0.11103) * cp
    m = intensity - F32(0.00860904) * ct - F32(0.11103) * cp
    s = intensity + F32(0.560031) * ct - F32(0.320627) * cp

    l_linear = eotf_st2084(l)
    m_linear = eotf_st2084(m)
    s_linear = eotf_st2084(s)

    r = np.maximum(
        F32(3.43661) * l_linear - F32(2.50645) * m_linear + F32(0.0698454) * s_linear,
        F32(0.0),
    )
    g = np.maximum(
        -F32(0.79133) * l_linear + F32(1.9836) * m_linear - F32(0.192271) * s_linear,
        F32(0.0),
    )
    b = np.maximum(
        -F32(0.0259499) * l_linear - F32(0.0989137) * m_linear + F32(1.12486) * s_linear,
        F32(0.0),
    )
    return np.stack((r, g, b), axis=-1).astype(np.float32)


@dataclass(frozen=True)
class GTToneMappingCurveV2:
    peak_intensity: np.float32
    alpha: np.float32
    mid_point: np.float32
    linear_section: np.float32
    toe_strength: np.float32
    k_a: np.float32
    k_b: np.float32
    k_c: np.float32

    @classmethod
    def create(
        cls,
        monitor_intensity: object,
        alpha: object,
        gray_point: object,
        linear_section: object,
        toe_strength: object,
    ) -> "GTToneMappingCurveV2":
        peak = F32(monitor_intensity)
        alpha = F32(alpha)
        mid_point = F32(gray_point)
        linear_section = F32(linear_section)
        toe_strength = F32(toe_strength)
        k = (linear_section - F32(1.0)) / (alpha - F32(1.0))
        k_a = peak * linear_section + peak * k
        k_b = -peak * k * np.exp(linear_section / k)
        k_c = -F32(1.0) / (k * peak)
        return cls(peak, alpha, mid_point, linear_section, toe_strength, k_a, k_b, k_c)

    def evaluate(self, x: object) -> Array:
        x = _f32(x)
        non_negative = np.maximum(x, F32(0.0))
        weight_linear = smooth_step(non_negative, F32(0.0), self.mid_point)
        weight_toe = F32(1.0) - weight_linear
        shoulder = self.k_a + self.k_b * np.exp(non_negative * self.k_c)
        toe_mapped = self.mid_point * np.power(
            non_negative / self.mid_point, self.toe_strength
        )
        below_shoulder = weight_toe * toe_mapped + weight_linear * non_negative
        result = np.where(
            non_negative < self.linear_section * self.peak_intensity,
            below_shoulder,
            shoulder,
        )
        return np.where(x < F32(0.0), F32(0.0), result).astype(np.float32)


@dataclass(frozen=True)
class GT7ToneMapping:
    """Fixed-parameter GT7 tone mapper using the sample's default ICtCp UCS."""

    sdr_correction_factor: np.float32
    framebuffer_luminance_target: np.float32
    framebuffer_luminance_target_ucs: np.float32
    curve: GTToneMappingCurveV2
    blend_ratio: np.float32 = F32(0.6)
    fade_start: np.float32 = F32(0.98)
    fade_end: np.float32 = F32(1.16)

    @classmethod
    def _create(cls, physical_target_luminance: object, sdr_correction: object) -> "GT7ToneMapping":
        target = F32(physical_target_luminance) / REFERENCE_LUMINANCE
        curve = GTToneMappingCurveV2.create(target, 0.25, 0.538, 0.444, 1.280)
        target_rgb = np.full((1, 3), target, dtype=np.float32)
        target_ucs = F32(rgb_to_ictcp(target_rgb)[0, 0])
        return cls(F32(sdr_correction), target, target_ucs, curve)

    @classmethod
    def sdr(cls) -> "GT7ToneMapping":
        correction = F32(1.0) / (GRAN_TURISMO_SDR_PAPER_WHITE / REFERENCE_LUMINANCE)
        return cls._create(GRAN_TURISMO_SDR_PAPER_WHITE, correction)

    @classmethod
    def hdr(cls, peak_luminance_nits: object = 1000.0) -> "GT7ToneMapping":
        return cls._create(peak_luminance_nits, F32(1.0))

    def apply(self, rgb: object) -> Array:
        """Apply tone mapping to non-negative linear Rec.2020 RGB values."""

        unclipped = self.apply_unclipped(rgb)
        output_peak = self.sdr_correction_factor * self.framebuffer_luminance_target
        return np.minimum(unclipped, output_peak).astype(np.float32)

    def apply_unclipped(self, rgb: object) -> Array:
        """Apply every GT7 step except the final separable per-channel minimum.

        Separating this hard minimum lets a following 1D LUT reproduce it exactly,
        while the 3D LUT interpolates the smoother cross-channel transform.
        """

        rgb = _f32(rgb)
        if rgb.shape[-1:] != (3,):
            raise ValueError("RGB input must have a final dimension of length 3")
        ucs = rgb_to_ictcp(rgb)
        skewed_rgb = self.curve.evaluate(rgb)
        skewed_ucs = rgb_to_ictcp(skewed_rgb)
        chroma_scale = chroma_curve(
            ucs[..., 0] / self.framebuffer_luminance_target_ucs,
            self.fade_start,
            self.fade_end,
        )
        scaled_ucs = np.stack(
            (
                skewed_ucs[..., 0],
                ucs[..., 1] * chroma_scale,
                ucs[..., 2] * chroma_scale,
            ),
            axis=-1,
        )
        scaled_rgb = ictcp_to_rgb(scaled_ucs)
        blended = (F32(1.0) - self.blend_ratio) * skewed_rgb + self.blend_ratio * scaled_rgb
        return (self.sdr_correction_factor * blended).astype(np.float32)


def normalize_mode(mode: str) -> str:
    normalized = mode.lower().replace("-", "").replace("_", "")
    if normalized == "sdr":
        return "sdr"
    if normalized == "hdr":
        return "hdr1000"
    if normalized.startswith("hdr"):
        try:
            peak = float(normalized[3:])
        except ValueError as error:
            raise ValueError(f"Unsupported mode: {mode!r}") from error
        if not np.isfinite(peak) or not 250.0 <= peak <= 10000.0:
            raise ValueError("HDR peak luminance must be in the range 250 to 10000 nits")
        return f"hdr{peak:g}"
    raise ValueError(f"Unsupported mode: {mode!r}")


def tone_mapper(mode: str) -> GT7ToneMapping:
    normalized = normalize_mode(mode)
    if normalized == "sdr":
        return GT7ToneMapping.sdr()
    return GT7ToneMapping.hdr(float(normalized[3:]))
