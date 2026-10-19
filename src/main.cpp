// 大気散乱 検証アプリ（docs/design.md）
//
// 使い方:
//   atmosphere.exe [--debug]                      対話モード
//   atmosphere.exe --test <name> --out <dir>      バッチ検証（Python の reference/validate_phase*.py から起動）
//     Phase 1: gt7, pq   Phase 2: transmittance, single_scattering
//     Phase 3: multiscattering, sky_lut, sun_closeup, pathtrace, spectral
//     Phase 4: aerial_perspective, sky_irradiance, perf（perf は Release で実行する）
//     Phase 4.6: ap_transmittance
//     Phase 4.7: pathtrace_aerosol（媒質は reference/validate_v4b.py が書き出す）、pathtrace_aerosol_views
//     Phase 4.5: shadow_surface, shadow_aerial_perspective, shadow_ap_methods
//     Phase 4.8: aerosol_presets（比較は reference/validate_phase4_8.py）
//     Phase 4.9: aerosol_presets（2 成分のプリセットを含む）、aerosol_scene（比較は reference/validate_phase4_9.py）
#include "app/App.h"
#include "batch/Phase1Tests.h"
#include "batch/Phase2Tests.h"
#include "batch/Phase3Tests.h"
#include "batch/Phase45Tests.h"
#include "batch/Phase48Tests.h"
#include "batch/Phase4Tests.h"

#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    bool debugLayer = false;
    std::string test;
    std::string outDir = "dumps";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--debug") {
            debugLayer = true;
        } else if (arg == "--test" && i + 1 < argc) {
            test = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            outDir = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 2;
        }
    }

    try {
        if (!test.empty()) {
            gfx::Device device(debugLayer);
            gfx::PipelineLibrary pipelines(device, ATMOS_SHADER_DIR);
            if (test == "gt7" || test == "pq") {
                return batch::runPhase1Test(device, pipelines, test, outDir);
            }
            if (test == "transmittance" || test == "single_scattering") {
                return batch::runPhase2Test(device, pipelines, test, outDir);
            }
            if (test == "aerial_perspective" || test == "ap_transmittance" || test == "sky_irradiance" ||
                test == "perf") {
                return batch::runPhase4Test(device, pipelines, test, outDir);
            }
            if (test == "shadow_surface" || test == "shadow_aerial_perspective" || test == "shadow_ap_methods") {
                return batch::runPhase45Test(device, pipelines, test, outDir);
            }
            if (test == "aerosol_presets" || test == "aerosol_scene") {
                return batch::runPhase48Test(device, pipelines, test, outDir);
            }
            return batch::runPhase3Test(device, pipelines, test, outDir);
        }
        app::App application(debugLayer);
        return application.run();
    } catch (const std::exception& e) {
        gfx::log(std::string("Fatal: ") + e.what());
        return 1;
    }
}
