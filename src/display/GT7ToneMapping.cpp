#include "display/GT7ToneMapping.h"

#include <glm/common.hpp>

#include <algorithm>
#include <cmath>

namespace display {
namespace {

constexpr float kAlpha = 0.25f;
constexpr float kMidPoint = 0.538f;
constexpr float kLinearSection = 0.444f;
constexpr float kToeStrength = 1.280f;
constexpr float kBlendRatio = 0.6f;
constexpr float kFadeStart = 0.98f;
constexpr float kFadeEnd = 1.16f;

// PQ 定数（SMPTE ST 2084）
constexpr float kPqM1 = 0.1593017578125f;
constexpr float kPqM2 = 78.84375f;
constexpr float kPqC1 = 0.8359375f;
constexpr float kPqC2 = 18.8515625f;
constexpr float kPqC3 = 18.6875f;

float smoothStep(float x, float edge0, float edge1)
{
    const float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// fb 値（1.0 = 100 nit）→ PQ 信号値。GT7 の UCS 変換で使う。
float inverseEotfFb(float fb)
{
    const float y = std::max(fb * kGTReferenceLuminanceNits / 10000.0f, 0.0f);
    const float ym = std::pow(y, kPqM1);
    return std::exp2(kPqM2 * (std::log2(kPqC1 + kPqC2 * ym) - std::log2(1.0f + kPqC3 * ym)));
}

// PQ 信号値 → fb 値。
float eotfFb(float n)
{
    n = std::clamp(n, 0.0f, 1.0f);
    const float np = std::pow(n, 1.0f / kPqM2);
    const float l = std::pow(std::max(np - kPqC1, 0.0f) / (kPqC2 - kPqC3 * np), 1.0f / kPqM1);
    return l * 10000.0f / kGTReferenceLuminanceNits;
}

glm::vec3 rgbToICtCp(const glm::vec3& rgb)
{
    const float l = (rgb.x * 1688.0f + rgb.y * 2146.0f + rgb.z * 262.0f) / 4096.0f;
    const float m = (rgb.x * 683.0f + rgb.y * 2951.0f + rgb.z * 462.0f) / 4096.0f;
    const float s = (rgb.x * 99.0f + rgb.y * 309.0f + rgb.z * 3688.0f) / 4096.0f;
    const float lp = inverseEotfFb(l);
    const float mp = inverseEotfFb(m);
    const float sp = inverseEotfFb(s);
    return {(2048.0f * lp + 2048.0f * mp) / 4096.0f,
            (6610.0f * lp - 13613.0f * mp + 7003.0f * sp) / 4096.0f,
            (17933.0f * lp - 17390.0f * mp - 543.0f * sp) / 4096.0f};
}

glm::vec3 iCtCpToRgb(const glm::vec3& v)
{
    const float l = eotfFb(v.x + 0.00860904f * v.y + 0.11103f * v.z);
    const float m = eotfFb(v.x - 0.00860904f * v.y - 0.11103f * v.z);
    const float s = eotfFb(v.x + 0.560031f * v.y - 0.320627f * v.z);
    return {std::max(3.43661f * l - 2.50645f * m + 0.0698454f * s, 0.0f),
            std::max(-0.79133f * l + 1.9836f * m - 0.192271f * s, 0.0f),
            std::max(-0.0259499f * l - 0.0989137f * m + 1.12486f * s, 0.0f)};
}

float curve(const GT7Constants& c, float x)
{
    x = std::max(x, 0.0f);
    const float weightLinear = smoothStep(x, 0.0f, c.midPoint);
    const float toe = c.midPoint * std::pow(std::min(x, c.midPoint) / c.midPoint, c.toeStrength);
    const float lower = (1.0f - weightLinear) * toe + weightLinear * x;
    if (x < c.linearSection * c.targetFb) {
        return lower;
    }
    return c.kA - c.targetFb * c.k * std::exp((c.linearSection - x / c.targetFb) / c.k);
}

} // namespace

GT7Constants makeGT7Constants(GT7Mode mode, float peakNits)
{
    GT7Constants c{};
    const float targetNits = mode == GT7Mode::SDR ? kGTSdrPaperWhiteNits
                                                  : std::clamp(peakNits, kGTMinPeakNits, kGTMaxPeakNits);
    c.targetFb = targetNits / kGTReferenceLuminanceNits;
    // [POINT] SDR では 250 nit を白 (1.0) に写すため 1/2.5 を掛ける。HDR では fb のまま出力し、
    //         表示段で ×100 して nit にしてから PQ エンコードする。
    c.sdrCorrection = mode == GT7Mode::SDR ? 1.0f / c.targetFb : 1.0f;
    c.midPoint = kMidPoint;
    c.linearSection = kLinearSection;
    c.toeStrength = kToeStrength;
    c.k = (kLinearSection - 1.0f) / (kAlpha - 1.0f);
    c.kA = c.targetFb * kLinearSection + c.targetFb * c.k;
    // [POINT] target の UCS 輝度はピーク輝度だけで決まる定数なので、ピクセルごとではなく
    //         ここで 1 回だけ計算する（PQ を 3 回評価する重い処理）。
    c.targetUcsI = rgbToICtCp(glm::vec3(c.targetFb)).x;
    c.blendRatio = kBlendRatio;
    c.fadeStart = kFadeStart;
    c.fadeEnd = kFadeEnd;
    return c;
}

glm::vec3 applyGT7(const GT7Constants& c, const glm::vec3& rec2020Fb)
{
    const glm::vec3 rgb = glm::max(rec2020Fb, glm::vec3(0.0f));
    const glm::vec3 ucs = rgbToICtCp(rgb);
    const glm::vec3 skewedRgb(curve(c, rgb.x), curve(c, rgb.y), curve(c, rgb.z));
    const glm::vec3 skewedUcs = rgbToICtCp(skewedRgb);
    const float chromaScale = 1.0f - smoothStep(ucs.x / c.targetUcsI, c.fadeStart, c.fadeEnd);
    const glm::vec3 scaledRgb = iCtCpToRgb(glm::vec3(skewedUcs.x, ucs.y * chromaScale, ucs.z * chromaScale));
    const glm::vec3 blended = (1.0f - c.blendRatio) * skewedRgb + c.blendRatio * scaledRgb;
    return c.sdrCorrection * glm::min(blended, glm::vec3(c.targetFb));
}

float pqEncodeNits(float nits)
{
    const float y = std::clamp(nits / 10000.0f, 0.0f, 1.0f);
    const float ym = std::pow(y, kPqM1);
    return std::pow((kPqC1 + kPqC2 * ym) / (1.0f + kPqC3 * ym), kPqM2);
}

} // namespace display
