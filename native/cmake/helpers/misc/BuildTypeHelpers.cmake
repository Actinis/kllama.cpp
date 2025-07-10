if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(IS_RELEASE ON)
    set(IS_DEBUG OFF)

    add_definitions(-DIS_RELEASE_BUILD=1)
else ()
    set(IS_RELEASE OFF)
    set(IS_DEBUG ON)

    add_definitions(-DIS_DEBUG_BUILD=1)
endif ()