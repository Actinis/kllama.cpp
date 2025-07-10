cmake_minimum_required(VERSION 3.31)
project(kllama_cpp_native)

#
# General
#
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (ANDROID)
    set(CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS "$ENV{ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/clang-scan-deps")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -frtti -std=c++20")
    set(CMAKE_ANDROID_STL_TYPE c++_shared)
endif ()

set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

include(cmake/helpers/misc/BuildTypeHelpers.cmake)
include(cmake/helpers/cpu/CPUHelpers.cmake)
include(cmake/helpers/platform/PlatformHelpers.cmake)

#
# Dependencies
#

add_subdirectory("${CMAKE_SOURCE_DIR}/logging" "${CMAKE_BINARY_DIR}/logging")
add_subdirectory("${CMAKE_SOURCE_DIR}/wrappers/llama.cpp" "${CMAKE_BINARY_DIR}/wrappers/llama.cpp")

#
# Library
#

add_subdirectory("${CMAKE_SOURCE_DIR}/library" "${CMAKE_BINARY_DIR}/library")
