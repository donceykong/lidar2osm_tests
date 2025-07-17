cmake_minimum_required(VERSION 3.24)
project(lidar2osm_tests)

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra -Wpedantic)
# endif()

# Set a default build type if none is specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    # Provide options for cmake-gui or ccmake
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

include(FetchContent)

# ==========================  Dependencies ==========================

# Shared Deps
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
# find_package(pcl_conversions REQUIRED)
# find_package(Eigen REQUIRED)

# KISS-Matcher-specific Dependencies
# find_package(teaserpp REQUIRED)
### To include ROBIN& KISS-Matcher ###
find_package(robin REQUIRED)
find_package(kiss_matcher REQUIRED)

# Lidar2OSM-specific Dependencies
find_package(clipper REQUIRED)

# ========================== Headers ==========================

# The best way to set proper compiler settings for using OpenMP in all platforms
include(FindOpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else (OPENMP_FOUND)
    message("ERROR: OpenMP could not be found.")
endif (OPENMP_FOUND)

option(USE_SYSTEM_TBB "Use system pre-installed oneAPI/tbb" ON)
include(3rdparty/find_dependencies.cmake)
include(3rdparty/download_datasets.cmake)

# Headers
include_directories(
    ${EIGEN3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    include
    src
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# ========================== Executables ==========================

add_executable(KISSMatcher src/KISSMatcher.cc)
target_include_directories(KISSMatcher
    PUBLIC
    ${PCL_INCLUDE_DIRS}
)
target_link_libraries(KISSMatcher
    Eigen3::Eigen
    TBB::tbb
    kiss_matcher::kiss_matcher_core
    robin::robin
    ${PCL_LIBRARIES}
)

add_executable(Lidar2OSM src/Lidar2OSM.cc)
target_include_directories(Lidar2OSM
    PUBLIC
    clipper
    ${PCL_INCLUDE_DIRS}
)
target_link_libraries(Lidar2OSM
    Eigen3::Eigen
    clipper
    TBB::tbb
    kiss_matcher::kiss_matcher_core
    # robin::robin
    ${PCL_LIBRARIES}
)