cmake_minimum_required(VERSION 3.25)

# vcpkg のツールチェーンを自動検出する（VCPKG_ROOT → %USERPROFILE%\vcpkg の順）。
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(_vcpkg_toolchain_candidates)
  if(DEFINED ENV{VCPKG_ROOT})
    list(APPEND _vcpkg_toolchain_candidates "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
  endif()
  if(DEFINED ENV{USERPROFILE})
    list(APPEND _vcpkg_toolchain_candidates "$ENV{USERPROFILE}/vcpkg/scripts/buildsystems/vcpkg.cmake")
  endif()
  foreach(_vcpkg_toolchain IN LISTS _vcpkg_toolchain_candidates)
    if(EXISTS "${_vcpkg_toolchain}")
      set(CMAKE_TOOLCHAIN_FILE "${_vcpkg_toolchain}" CACHE FILEPATH "vcpkg toolchain file")
      message(STATUS "Using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
      break()
    endif()
  endforeach()
endif()

project(atmosphere CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(glm CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(tinyexr CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

# ---------------------------------------------------------------------------
# シェーダー: Windows SDK 同梱の DXC でビルド時にコンパイルし、.cso を実行時に読む。
# SM 6.6 の ResourceDescriptorHeap（Bindless）を使うため 6.6 以上が必要。
# ---------------------------------------------------------------------------
find_program(DXC_EXECUTABLE NAMES dxc
  HINTS "$ENV{WindowsSdkVerBinPath}/x64"
        "D:/Windows Kits/10/bin/10.0.26100.0/x64"
        "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64"
  REQUIRED)
message(STATUS "DXC: ${DXC_EXECUTABLE}")

set(SHADER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

# インクルードファイルの変更で全シェーダーを再コンパイルする（依存解析を持たない代わりの単純な方法）。
file(GLOB_RECURSE SHADER_HEADERS CONFIGURE_DEPENDS ${SHADER_SRC_DIR}/*.hlsli)

set(SHADER_OUTPUTS)
# 追加の引数は DXC にそのまま渡す（例: -D MIE_GENERAL=1）
function(compile_shader source entry profile output_name)
  set(src ${SHADER_SRC_DIR}/${source})
  set(out ${SHADER_OUT_DIR}/${output_name}.cso)
  add_custom_command(
    OUTPUT ${out}
    COMMAND ${DXC_EXECUTABLE} -nologo -HV 2021 -WX -Zi -Qembed_debug
            -I ${SHADER_SRC_DIR} -T ${profile} -E ${entry} ${ARGN} -Fo ${out} ${src}
    DEPENDS ${src} ${SHADER_HEADERS}
    COMMENT "dxc ${source} ${entry} (${profile})"
    VERBATIM)
  set(SHADER_OUTPUTS ${SHADER_OUTPUTS} ${out} PARENT_SCOPE)
endfunction()

compile_shader(display/Display.hlsl VSMain vs_6_6 Display_vs)
compile_shader(display/Display.hlsl PSMain ps_6_6 Display_ps)
compile_shader(scene/TestPattern.hlsl CSMain cs_6_6 TestPattern_cs)
compile_shader(atmosphere/TransmittanceLut.hlsl CSMain cs_6_6 TransmittanceLut_cs)
compile_shader(atmosphere/SkyRaymarch.hlsl CSMain cs_6_6 SkyRaymarch_cs)
compile_shader(atmosphere/MultiScatteringLut.hlsl CSMain cs_6_6 MultiScatteringLut_cs)
compile_shader(atmosphere/SkyViewLut.hlsl CSMain cs_6_6 SkyViewLut_cs)
compile_shader(atmosphere/SkyFromLut.hlsl CSMain cs_6_6 SkyFromLut_cs)
compile_shader(atmosphere/PathTrace.hlsl CSMain cs_6_6 PathTrace_cs)
compile_shader(atmosphere/AerialPerspective.hlsl CSMain cs_6_6 AerialPerspective_cs)
compile_shader(atmosphere/SkyIrradiance.hlsl CSMain cs_6_6 SkyIrradiance_cs)
# Mie の一般形（2 成分・自由大気の背景、design.md §5.9）の版。1 成分・背景なしのプリセットは上の版を使う
# （成分 1 の分岐を残すと 1 成分でも GPU 時間が増えたため。AtmosphereCommon.hlsli の MIE_GENERAL）
foreach(pass TransmittanceLut SkyRaymarch MultiScatteringLut SkyViewLut SkyFromLut PathTrace AerialPerspective
             SkyIrradiance)
  compile_shader(atmosphere/${pass}.hlsl CSMain cs_6_6 ${pass}_mie_cs -D MIE_GENERAL=1)
endforeach()
# 多重散乱の SH 近似（design.md §5.11）の版。多重散乱を扱う 4 つのパスだけ。MS_SH = 0 の版は上と同じ
# （AtmosphereCommon.hlsli の MS_SH）
foreach(pass SkyRaymarch MultiScatteringLut SkyViewLut AerialPerspective)
  compile_shader(atmosphere/${pass}.hlsl CSMain cs_6_6 ${pass}_sh_cs -D MS_SH=1)
  compile_shader(atmosphere/${pass}.hlsl CSMain cs_6_6 ${pass}_mie_sh_cs -D MIE_GENERAL=1 -D MS_SH=1)
endforeach()
# SH の Multi-scattering LUT の F の表（行 = 高度、天頂角の 1 次元の求積）。係数のパスより先に回す
compile_shader(atmosphere/MultiScatteringLut.hlsl CSTransfer cs_6_6 MultiScatteringShTransfer_sh_cs -D MS_SH=1)
compile_shader(atmosphere/MultiScatteringLut.hlsl CSTransfer cs_6_6 MultiScatteringShTransfer_mie_sh_cs
               -D MIE_GENERAL=1 -D MS_SH=1)
compile_shader(scene/Mesh.hlsl VSMain vs_6_6 Mesh_vs)
compile_shader(scene/Mesh.hlsl PSMain ps_6_6 Mesh_ps)

add_custom_target(shaders DEPENDS ${SHADER_OUTPUTS} SOURCES ${SHADER_HEADERS})

# ---------------------------------------------------------------------------
# アプリ本体
# ---------------------------------------------------------------------------
add_executable(atmosphere
  src/main.cpp
  src/app/App.cpp
  src/app/Window.cpp
  src/atmosphere/AtmosphereParams.cpp
  src/atmosphere/AtmosphereRenderer.cpp
  src/atmosphere/MiePhaseTables.cpp
  src/atmosphere/PhaseMoments.cpp
  src/atmosphere/PathTracer.cpp
  src/gfx/Device.cpp
  src/gfx/Buffer.cpp
  src/gfx/Descriptors.cpp
  src/gfx/GpuTimer.cpp
  src/gfx/Pipeline.cpp
  src/gfx/Texture.cpp
  src/display/ColorSpace.cpp
  src/display/DisplayMonitor.cpp
  src/display/DisplayPass.cpp
  src/display/GT7ToneMapping.cpp
  src/display/Swapchain.cpp
  src/scene/Camera.cpp
  src/scene/Meshes.cpp
  src/scene/TestPatternPass.cpp
  src/render/SceneRenderer.cpp
  src/render/ShadowRenderer.cpp
  src/ui/ImGuiLayer.cpp
  src/io/ExrWriter.cpp
  src/batch/Phase1Tests.cpp
  src/batch/Phase2Tests.cpp
  src/batch/Phase3Tests.cpp
  src/batch/Phase4Tests.cpp
  src/batch/Phase45Tests.cpp
  src/batch/Phase48Tests.cpp
  src/batch/Phase411Tests.cpp
)
add_dependencies(atmosphere shaders)

target_include_directories(atmosphere PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(atmosphere PRIVATE
  UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN
  GLM_FORCE_XYZW_ONLY
  ATMOS_SHADER_DIR="${SHADER_OUT_DIR}"
  ATMOS_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
# /utf-8: ソースは BOM なし UTF-8（日本語コメント）なので明示する。
target_compile_options(atmosphere PRIVATE /W4 /permissive- /utf-8 /Zc:__cplusplus)
target_link_libraries(atmosphere PRIVATE
  d3d12 dxgi dxguid
  glm::glm
  nlohmann_json::nlohmann_json
  unofficial::tinyexr::tinyexr
  imgui::imgui)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT atmosphere)
