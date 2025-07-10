add_library(logging INTERFACE)

target_sources(logging INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/logging/logging.h
)

target_include_directories(logging INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (ANDROID)
    find_library(log-lib log)
    target_compile_definitions(logging INTERFACE ANDROID)
    target_include_directories(logging INTERFACE ${ANDROID_SYSROOT}/usr/include)
    target_link_libraries(logging INTERFACE ${log-lib})
elseif (LINUX OR WINDOWS OR IOS)
    set(SPDLOG_BUILD_PIC ON CACHE BOOL "" FORCE)
    set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
    set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)

    add_subdirectory(${CMAKE_SOURCE_DIR}/external/spdlog ${CMAKE_BINARY_DIR}/external/spdlog)
    target_link_libraries(logging INTERFACE spdlog::spdlog)
endif ()

#
# Demo
#

add_executable(logging_demo
        src/demo/main.cpp
)

add_dependencies(logging_demo
        logging
)

target_link_libraries(logging_demo
        logging
)

target_include_directories(logging_demo PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)