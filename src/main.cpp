// 大気散乱 検証アプリ（docs/design.md）
//
// 使い方:
//   atmosphere.exe [--debug]                      対話モード
//   atmosphere.exe --test <gt7|pq> --out <dir>    バッチ検証（Python の reference/validate_phase1.py から起動）
#include "app/App.h"
#include "batch/Phase1Tests.h"

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
            return batch::runPhase1Test(device, pipelines, test, outDir);
        }
        app::App application(debugLayer);
        return application.run();
    } catch (const std::exception& e) {
        gfx::log(std::string("Fatal: ") + e.what());
        return 1;
    }
}
