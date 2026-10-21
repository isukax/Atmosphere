#pragma once
// Test-only readback; g aliases the selected GPU backend before this header is included.
std::vector<uint8_t> readTexture(g::Device &device, g::Texture &texture, uint32_t bytesPerPixel) {
    D3D12_RESOURCE_DESC rd = texture.resource()->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 totalBytes = 0;
    device.d3d()->GetCopyableFootprints(&rd, 0, 1, 0, &footprint, nullptr, nullptr, &totalBytes);
    Microsoft::WRL::ComPtr<ID3D12Resource> staging =
        g::createBuffer(device.d3d(), totalBytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    device.executeImmediate([&](ID3D12GraphicsCommandList *cmd) {
        texture.transition(cmd, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION src{texture.resource(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        src.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION dst{staging.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        dst.PlacedFootprint = footprint;
        cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    });

    // 3D テクスチャはスライス z ごとに height 行（出力は z, y, x の順に詰める）
    const uint32_t depth = texture.desc().depth;
    std::vector<uint8_t> out(static_cast<size_t>(texture.desc().width) * texture.desc().height * depth * bytesPerPixel);
    uint8_t *mapped = nullptr;
    g::check(staging->Map(0, nullptr, reinterpret_cast<void **>(&mapped)), "Map(readback)");
    const uint32_t rowBytes = texture.desc().width * bytesPerPixel;
    for (uint32_t z = 0; z < depth; ++z) {
        for (uint32_t y = 0; y < texture.desc().height; ++y) {
            const size_t row = static_cast<size_t>(z) * texture.desc().height + y;
            std::memcpy(out.data() + row * rowBytes, mapped + footprint.Offset + row * footprint.Footprint.RowPitch,
                        rowBytes);
        }
    }
    staging->Unmap(0, nullptr);
    return out;
}
std::vector<uint8_t> readBuffer(g::Device &device, g::Buffer &buffer) {
    Microsoft::WRL::ComPtr<ID3D12Resource> staging = g::createBuffer(
        device.d3d(), buffer.desc().sizeBytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    device.executeImmediate([&](ID3D12GraphicsCommandList *cmd) {
        buffer.transition(cmd, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmd->CopyBufferRegion(staging.Get(), 0, buffer.resource(), 0, buffer.desc().sizeBytes);
    });
    std::vector<uint8_t> out(buffer.desc().sizeBytes);
    void *mapped = nullptr;
    g::check(staging->Map(0, nullptr, &mapped), "Map(buffer readback)");
    std::memcpy(out.data(), mapped, buffer.desc().sizeBytes);
    staging->Unmap(0, nullptr);
    return out;
}
