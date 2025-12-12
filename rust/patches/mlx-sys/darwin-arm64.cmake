# CMake toolchain file to force ARM64 on macOS
set(CMAKE_SYSTEM_NAME Darwin)
set(CMAKE_SYSTEM_PROCESSOR arm64)
set(CMAKE_OSX_ARCHITECTURES arm64)
set(CMAKE_C_COMPILER_TARGET arm64-apple-darwin)
set(CMAKE_CXX_COMPILER_TARGET arm64-apple-darwin)
set(CMAKE_ASM_COMPILER_TARGET arm64-apple-darwin)

