#include "atmosphere/MiePhaseTables.h"

#include "gfx/Descriptors.h"

#include <string>

namespace atmosphere {

static_assert(kNoMiePhaseTable == gfx::kInvalidDescriptorIndex);

MiePhaseTables::MiePhaseTables(gfx::Device& device, std::vector<CoefficientSet>& sets)
{
    for (CoefficientSet& set : sets) {
        for (AerosolPreset& preset : set.aerosolPresets) {
            for (MieComponentPreset& c : preset.components) {
                if (c.phaseTable.empty()) {
                    continue;
                }
                const uint64_t bytes = c.phaseTable.size() * sizeof(glm::vec4);
                gfx::Buffer buffer(device, {bytes, sizeof(glm::vec4), true, false, L"MiePhaseTable"});
                buffer.upload(device, c.phaseTable.data(), bytes, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
                c.phaseTableSrv = buffer.srvIndex();
                m_buffers.push_back(std::move(buffer));
            }
        }
    }
}

} // namespace atmosphere
