#
# Library
#

set(LLAMACPP_DIR "${CMAKE_SOURCE_DIR}/external/llama.cpp")

set(LIBRARY_HEADERS
        src/lib/KLlama.h
)

set(LIBRARY_SOURCES
        src/lib/KLlama.cpp
        ${LLAMACPP_DIR}/tools/mtmd/mtmd.cpp
        ${LLAMACPP_DIR}/tools/mtmd/mtmd-helper.cpp
        ${LLAMACPP_DIR}/tools/mtmd/mtmd-audio.cpp
        ${LLAMACPP_DIR}/tools/mtmd/clip.cpp
)

add_library(kllama_cpp_native STATIC
        ${LIBRARY_HEADERS}
        ${LIBRARY_SOURCES}
)

target_include_directories(kllama_cpp_native PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src/lib"
        "${LLAMACPP_DIR}/tools/mtmd"
        "${LLAMACPP_DIR}/vendor"
)

add_dependencies(kllama_cpp_native
        logging
        llama
)

target_link_libraries(kllama_cpp_native
        logging
        llama
)

#
# Demo
#

set(LIBRARY_DEMO_SOURCES
        src/demo/library_demo.cpp
)

add_executable(kllama_cpp_native_demo
        ${LIBRARY_DEMO_SOURCES}
)

target_link_libraries(kllama_cpp_native_demo
        kllama_cpp_native
)