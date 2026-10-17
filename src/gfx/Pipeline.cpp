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

    D3D12_STATIC_SAMPLER_DESC samplers[3]{};
    for (UINT i = 0; i < 3; ++i) {
        samplers[i].Filter = i == 1 ? D3D12_FILTER_MIN_MAG_MIP_POINT : D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        samplers[i].AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        samplers[i].AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        samplers[i].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        samplers[i].MaxLOD = D3D12_FLOAT32_MAX;
        samplers[i].ShaderRegister = i;
        samplers[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    // [POINT] s2: シャドウマップの比較サンプラー。シャドウマップの深度は光源に近いほど大きい（Reverse）ので、
    //         「受け手の深度 >= 遮蔽物の深度」で日なた（GREATER_EQUAL）。線形補間で 2×2 の PCF になる。
    samplers[2].Filter = D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
    samplers[2].ComparisonFunc = D3D12_COMPARISON_FUNC_GREATER_EQUAL;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    desc.Desc_1_1.NumParameters = 1;
    desc.Desc_1_1.pParameters = params;
    desc.Desc_1_1.NumStaticSamplers = 3;
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

ComPtr<ID3D12PipelineState> PipelineLibrary::createMesh(const char* vsName, const char* psName, DXGI_FORMAT rtvFormat,
                                                        DXGI_FORMAT dsvFormat) const
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
    // カリングなし: 検証用シーンは小さく、巻き順の取り違えで面が消える危険を避ける
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.DepthClipEnable = TRUE;
    desc.DepthStencilState.DepthEnable = TRUE;
    desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_GREATER;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = rtvFormat;
    desc.DSVFormat = dsvFormat;
    desc.SampleDesc.Count = 1;
    ComPtr<ID3D12PipelineState> pso;
    check(m_device.d3d()->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso)), psName);
    return pso;
}

ComPtr<ID3D12PipelineState> PipelineLibrary::createDepthOnly(const char* vsName, DXGI_FORMAT dsvFormat,
                                                             float slopeScaledDepthBias) const
{
    const std::vector<uint8_t> vs = load(vsName);
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = m_rootSignature.Get();
    desc.VS = {vs.data(), vs.size()};
    desc.SampleMask = UINT_MAX;
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.SlopeScaledDepthBias = slopeScaledDepthBias;
    desc.RasterizerState.DepthClipEnable = TRUE;
    desc.DepthStencilState.DepthEnable = TRUE;
    desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_GREATER;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = 0;
    desc.DSVFormat = dsvFormat;
    desc.SampleDesc.Count = 1;
    ComPtr<ID3D12PipelineState> pso;
    check(m_device.d3d()->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso)), vsName);
    return pso;
}

} // namespace gfx
