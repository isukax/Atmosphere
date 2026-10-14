#include "io/ExrWriter.h"

#include <tinyexr.h>

#include <fstream>
#include <stdexcept>

namespace io {

void saveExr(const std::filesystem::path& path, uint32_t width, uint32_t height, const std::vector<float>& rgba,
             const nlohmann::json& meta)
{
    if (rgba.size() != static_cast<size_t>(width) * height * 4) {
        throw std::runtime_error("saveExr: size mismatch");
    }
    std::filesystem::create_directories(path.parent_path());
    const char* error = nullptr;
    // save_as_fp16 = 0: float32 のまま保存する（検証値を半精度で丸めない）。
    const int result = SaveEXR(rgba.data(), static_cast<int>(width), static_cast<int>(height), 4, 0,
                               path.string().c_str(), &error);
    if (result != TINYEXR_SUCCESS) {
        const std::string message = error ? error : "unknown";
        FreeEXRErrorMessage(error);
        throw std::runtime_error("SaveEXR failed: " + message);
    }

    nlohmann::json sidecar = meta;
    sidecar["schema_version"] = 1;
    sidecar["width"] = width;
    sidecar["height"] = height;
    std::filesystem::path metaPath = path;
    metaPath.replace_extension(".meta.json");
    std::ofstream(metaPath) << sidecar.dump(2) << "\n";
}

} // namespace io
