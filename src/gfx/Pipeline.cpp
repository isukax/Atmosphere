#include "gfx/Pipeline.h"

#include <fstream>

namespace gfx {

PipelineLibrary::PipelineLibrary(Device& device, std::filesystem::path shaderDir)
    : m_device(device), m_shaderDir(std::move(shaderDir))
{
    D3D12_ROOT_PARAMETER1 params[1]{};
    params[kRootPassConstants].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[kRootPassConstants].Descriptor.ShaderRegister = 0;
    params[kRootPassConstants].Descriptor.Flags = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE;
    params[kRootPassConstants].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_STATIC_SAMPLER_DESC samplers[2]{};
    for (UINT i = 0; i < 2; ++i) {
        samplers[i].Filter = i == 0 ? D3D12_FILTER_MIN_MAG_MIP_LINEAR : D3D12_FILTER_MIN_MAG_MIP_POINT;
        samplers[i].AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        samplers[i].AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        samplers[i].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        samplers[i].MaxLOD = D3D12_FLOAT32_MAX;
        samplers[i].ShaderRegister = i;
        samplers[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    desc.Desc_1_1.NumParameters = 1;
    desc.Desc_1_1.pParameters = params;
    desc.Desc_1_1.NumStaticSamplers = 2;
    desc.Desc_1_1.pStaticSamplers = samplers;
    // [POINT] このフラグが SM 6.6 の ResourceDescriptorHeap[] を有効にする。
    desc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED;

    ComPtr<ID3DBlob> blob;
    ComPtr<ID3DBlob> error;
    const HRESULT hr = D3D12SerializeVersionedRootSignature(&desc, &blob, &error);
    if (FAILED(hr)) {
        throw std::runtime_error(std::string("Root signature: ") +
                                 (error ? static_cast<const char*>(error->GetBufferPointer()) : "unknown"));
    }
    check(device.d3d()->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                            IID_PPV_ARGS(&m_rootSignature)),
          "CreateRootSignature");
}

std::vector<uint8_t> PipelineLibrary::load(const char* name) const
{
    const std::filesystem::path path = m_shaderDir / (std::string(name) + ".cso");
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Shader not found: " + path.string());
    }
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

ComPtr<ID3D12PipelineState> PipelineLibrary::createCompute(const char* csName) const
{
    const std::vector<uint8_t> cs = load(csName);
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = m_rootSignature.Get();
    desc.CS = {cs.data(), cs.size()};
    ComPtr<ID3D12PipelineState> pso;
    check(m_device.d3d()->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)), csName);
    return pso;
}

ComPtr<ID3D12PipelineState> PipelineLibrary::createFullscreen(const char* vsName, const char* psName,
                                                              DXGI_FORMAT rtvFormat) const
{
    const std::vector<uint8_t> vs = load(vsName);
    const std::vector<uint8_t> ps = load(psName);
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = m_rootSignature.Get();
    desc.VS = {vs.data(), vs.size()};
    desc.PS = {ps.data(), ps.size()};
    desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    desc.SampleMask = UINT_MAX;
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = rtvFormat;
    desc.SampleDesc.Count = 1;
    ComPtr<ID3D12PipelineState> pso;
    check(m_device.d3d()->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso)), psName);
    return pso;
}

} // namespace gfx
