if (ANDROID)
    include(${CMAKE_SOURCE_DIR}/cmake/helpers/platform/AndroidHelpers.cmake)
elseif (IOS)
    include(${CMAKE_SOURCE_DIR}/cmake/helpers/platform/IOSHelpers.cmake)
endif ()

#add_definitions(-DACTINIS_SKIA_DARWIN=1) # To test on non-mac