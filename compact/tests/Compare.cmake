# Optional repository-only tools. The standalone library/sample do not use these paths.
set(COMPACT_REFERENCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH "Original verification project")
if(NOT EXISTS "${COMPACT_REFERENCE_ROOT}/src/atmosphere/AtmosphereParams.cpp")
  message(FATAL_ERROR "COMPACT_BUILD_COMPARE requires the original verification project")
endif()
find_package(nlohmann_json CONFIG REQUIRED)
add_executable(compact_dump tests/Dump.cpp)
target_link_libraries(compact_dump PRIVATE compact_scene)
target_compile_definitions(compact_dump PRIVATE COMPACT_SHADER_DIR="${COMPACT_SHADER_DIR}")
target_compile_options(compact_dump PRIVATE /W4 /permissive- /utf-8)

# Compile original sources as-is; original shader binaries are read from its existing build.
set(COMPACT_REFERENCE_SHADERS "${COMPACT_REFERENCE_ROOT}/build/vs2026-x64/shaders" CACHE PATH "Original CSO directory")
set(reference_sources gfx/Device.cpp gfx/Buffer.cpp gfx/Descriptors.cpp gfx/Pipeline.cpp gfx/Texture.cpp
  atmosphere/AtmosphereParams.cpp atmosphere/AtmosphereRenderer.cpp atmosphere/PhaseMoments.cpp atmosphere/MiePhaseTables.cpp
  render/SceneRenderer.cpp render/ShadowRenderer.cpp scene/Camera.cpp scene/Meshes.cpp display/ColorSpace.cpp)
list(TRANSFORM reference_sources PREPEND "${COMPACT_REFERENCE_ROOT}/src/")
add_executable(compact_reference_dump tests/Dump.cpp ${reference_sources})
target_include_directories(compact_reference_dump PRIVATE "${COMPACT_REFERENCE_ROOT}/src")
target_link_libraries(compact_reference_dump PRIVATE d3d12 dxgi dxguid glm::glm nlohmann_json::nlohmann_json)
target_compile_options(compact_reference_dump PRIVATE /W4 /permissive- /utf-8)
target_compile_definitions(compact_reference_dump PRIVATE COMPACT_REFERENCE UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN GLM_FORCE_XYZW_ONLY
  COMPACT_SHADER_DIR="${COMPACT_REFERENCE_SHADERS}" COMPACT_COEFFICIENTS="${COMPACT_REFERENCE_ROOT}/data/atmosphere_coefficients.json")
