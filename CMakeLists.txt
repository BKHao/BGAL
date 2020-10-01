cmake_minimum_required(VERSION 3.17)
project(BKHao_CODE_2)


# Default Build_type RELEASE
set(CMAKE_BUILD_TYPE RELEASE)
# CXX STANDARD
set(CMAKE_CXX_STANDARD 17)
# Set Version
set(VERSION 1.0)
# include header from project_source_dir
include_directories(${PROJECT_SOURCE_DIR}/include)

# Get Eigen3
find_package(Eigen3 REQUIRED)
if (Eigen3_FOUND)
    message(STATUS "${EIGEN3_VERSION_STRING}")
    include_directories(${EIGEN3_INCLUDE_DIR})
endif ()

# Get Boost
find_package(Boost REQUIRED)
if (Boost_FOUND)
    message(STATUS "BOOST FOUNDED")
    include_directories(${Boost_INCLUDE_DIRS})
endif ()

# Get CGAL
find_package(CGAL REQUIRED)
if (CGAL_FOUND)
    include(${CGAL_USE_FILE})
else ()
    message("ERROR: this program requires CGAL and will not be compiled.")
endif ()

# Set lib output directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# Generator to static?
option(BUILD_STATIC_LIB "build BKHao.a static lib" OFF)

add_subdirectory(include)
add_subdirectory(src)

target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})

