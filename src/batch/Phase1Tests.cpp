#include "batch/Phase1Tests.h"

#include "display/DisplayPass.h"
#include "gfx/Texture.h"
#include "io/ExrWriter.h"

#include <cmath>
#include <cstring>
#include <fstream>

namespace batch {
namespace {

nlohmann::json matrixJson(const display::Mat3& m)
{
    nlohmann::json rows = nlohmann::json::array();
    for (const auto& row : m.m) {
        rows.push_back({row[0], row[1], row[2]});
    }
    return rows;
}

std::vector<float> toFloats(const std::vector<uint8_t>& bytes)
{
    std::vector<float> out(bytes.size() / sizeof(float));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

/// 入力テクスチャ（float32 RGBA）をシーンカラーとして表示パスに通し、`format` の RT に描く。
gfx::Texture renderDisplay(gfx::Device& device, display::DisplayPass& pass, const gfx::Texture& input,
                           DXGI_FORMAT format, const display::DisplayPassInput& params)
{
    const uint32_t w = input.desc().width;
    const uint32_t h = input.desc().height;
    gfx::Texture target(device, {w, h, format, false, false, true, L"Phase1Target"});
    device.upload().beginSlot(0);
    device.executeImmediate([&](ID3D12GraphicsCommandList* cmd) {
        target.transition(cmd, D3D12_RESOURCE_STATE_RENDER_TARGET);
        pass.record(cmd, device.upload(), target.rtv(), format, w, h, params);
    });
    return target;
}

int runGT7(gfx::Device& device, display::DisplayPass& pass, const std::filesystem::path& outDir)
{
    // [POINT] 入力は linear Rec.2020 の fb 値の格子: 各チャンネル 33 値
    //         （0 と 10^-4〜10^2 fb の対数間隔 = 0.01 nit〜10,000 nit）の全組み合わせ 33³ 点。
    //         低輝度のトゥ、線形区間、ショルダー、クロマフェード、クリップ境界をすべて通る。
    constexpr uint32_t kN = 33;
    std::vector<float> values(kN);
    values[0] = 0.0f;
    for (uint32_t i = 1; i < kN; ++i) {
        values[i] = std::pow(10.0f, -4.0f + 6.0f * static_cast<float>(i - 1) / static_cast<float>(kN - 2));
    }
    const uint32_t w = kN * kN;
    const uint32_t h = kN;
    std::vector<float> inputPixels(static_cast<size_t>(w) * h * 4);
    for (uint32_t b = 0; b < kN; ++b) {
        for (uint32_t g = 0; g < kN; ++g) {
            for (uint32_t r = 0; r < kN; ++r) {
                float* p = &inputPixels[(static_cast<size_t>(b) * w + g * kN + r) * 4];
                p[0] = values[r];
                p[1] = values[g];
                p[2] = values[b];
                p[3] = 1.0f;
            }
        }
    }
    gfx::Texture input(device, {w, h, DXGI_FORMAT_R32G32B32A32_FLOAT, true, false, false, L"GT7Input"});
    input.upload(device, inputPixels.data(), 16);
    io::saveExr(outDir / "gt7_input.exr", w, h, inputPixels,
                {{"units", "fb (1.0 = 100 cd/m2)"}, {"color_space", "Rec2020 linear"}, {"role", "gt7 input"}});

    struct Mode {
        const char* name;
        display::GT7Mode mode;
        float peakNits;
    };
    const Mode modes[] = {
        {"sdr", display::GT7Mode::SDR, 0.0f},
        {"hdr450", display::GT7Mode::HDR, 450.0f},
        {"hdr1000", display::GT7Mode::HDR, 1000.0f},
        {"hdr2000", display::GT7Mode::HDR, 2000.0f},
    };
    for (const Mode& mode : modes) {
        display::DisplayPassInput params;
        params.sceneColorSrv = input.srvIndex();
        params.workingSpace = display::WorkingColorSpace::Rec2020;  // 単位行列: 入力をそのまま GT7 へ
        params.encode = display::OutputEncode::LinearRec2020;
        params.toneMap = display::ToneMapOperator::GT7;
        params.gt7 = display::makeGT7Constants(mode.mode, mode.peakNits);
        gfx::Texture target = renderDisplay(device, pass, input, DXGI_FORMAT_R32G32B32A32_FLOAT, params);
        // [POINT] 数値比較は float32 RT から読み戻す。FP16 だと格納時の丸め（AMD RDNA3 では切り捨て）で
        //         トーンマップではなく格納精度を測ることになる。
        const std::vector<float> out = toFloats(target.readback(device, 16));
        io::saveExr(outDir / (std::string("gt7_") + mode.name + ".exr"), w, h, out,
                    {{"units", mode.mode == display::GT7Mode::SDR ? "sdr relative (1.0 = SDR white)"
                                                                  : "fb (1.0 = 100 cd/m2)"},
                     {"color_space", "Rec2020 linear"},
                     {"role", "gt7 output"},
                     {"mode", mode.name},
                     {"peak_nits", mode.peakNits}});
        gfx::log(std::string("gt7: wrote ") + mode.name);
    }
    return 0;
}

int runPQ(gfx::Device& device, display::DisplayPass& pass, const std::filesystem::path& outDir)
{
    // 入力: 0 と 10^-3〜10^4 nit の対数間隔 1024 点（R）、G = R/2、B = R/4。fb = nit / 100。
    constexpr uint32_t kW = 1024;
    std::vector<float> inputPixels(kW * 4);
    for (uint32_t i = 0; i < kW; ++i) {
        const float nits = i == 0 ? 0.0f : std::pow(10.0f, -3.0f + 7.0f * static_cast<float>(i - 1) / (kW - 2));
        inputPixels[i * 4 + 0] = nits / 100.0f;
        inputPixels[i * 4 + 1] = nits * 0.5f / 100.0f;
        inputPixels[i * 4 + 2] = nits * 0.25f / 100.0f;
        inputPixels[i * 4 + 3] = 1.0f;
    }
    gfx::Texture input(device, {kW, 1, DXGI_FORMAT_R32G32B32A32_FLOAT, true, false, false, L"PQInput"});
    input.upload(device, inputPixels.data(), 16);
    io::saveExr(outDir / "pq_input.exr", kW, 1, inputPixels,
                {{"units", "fb (1.0 = 100 cd/m2)"}, {"color_space", "Rec2020 linear"}, {"role", "pq input"}});

    display::DisplayPassInput params;
    params.sceneColorSrv = input.srvIndex();
    params.workingSpace = display::WorkingColorSpace::Rec2020;
    params.encode = display::OutputEncode::Hdr10Pq;
    params.toneMap = display::ToneMapOperator::None;
    // 実際のスワップチェーンと同じ R10G10B10A2_UNORM に描き、整数コードで比較する。
    gfx::Texture target = renderDisplay(device, pass, input, DXGI_FORMAT_R10G10B10A2_UNORM, params);
    const std::vector<uint8_t> packed = target.readback(device, 4);
    std::vector<float> codes(kW * 4);
    for (uint32_t i = 0; i < kW; ++i) {
        uint32_t v = 0;
        std::memcpy(&v, &packed[i * 4], 4);
        // [POINT] DXGI_FORMAT_R10G10B10A2_UNORM のビット配置: R = 0-9, G = 10-19, B = 20-29, A = 30-31。
        codes[i * 4 + 0] = static_cast<float>(v & 0x3FF);
        codes[i * 4 + 1] = static_cast<float>((v >> 10) & 0x3FF);
        codes[i * 4 + 2] = static_cast<float>((v >> 20) & 0x3FF);
        codes[i * 4 + 3] = static_cast<float>(v >> 30);
    }
    io::saveExr(outDir / "pq_codes.exr", kW, 1, codes,
                {{"units", "PQ code (10bit integer, 0-1023)"}, {"color_space", "Rec2020 PQ"}, {"role", "pq output"}});
    gfx::log("pq: wrote codes");
    return 0;
}

} // namespace

int runPhase1Test(gfx::Device& device, const gfx::PipelineLibrary& pipelines, std::string_view name,
                  const std::filesystem::path& outDir)
{
    std::filesystem::create_directories(outDir);
    // 色空間行列も Python 側で colour-science の値と照合できるように書き出す。
    nlohmann::json matrices = {
        {"rec709_to_rec2020", matrixJson(display::kRec709ToRec2020)},
        {"rec2020_to_rec709", matrixJson(display::kRec2020ToRec709)},
    };
    std::ofstream(outDir / "matrices.json") << matrices.dump(2) << "\n";

    display::DisplayPass pass(pipelines);
    if (name == "gt7") {
        return runGT7(device, pass, outDir);
    }
    if (name == "pq") {
        return runPQ(device, pass, outDir);
    }
    gfx::log("Unknown test: " + std::string(name));
    return 2;
}

} // namespace batch
