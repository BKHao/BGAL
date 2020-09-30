cmake_minimum_required(VERSION 3.18)
project(BKHao_CODE_2)

find_package(Eigen3 REQUIRED)
if(Eigen3_FOUND)
message(STATUS "${EIGEN3_VERSION_STRING}")
include_directories(${EIGEN3_INCLUDE_DIR})
endif()

find_package(Boost REQUIRED)
if(Boost_FOUND)
message(STATUS "BOOST FOUNDED")
include_directories(${Boost_INCLUDE_DIRS}) 
endif()

find_package(CGAL REQUIRED)
if(CGAL_FOUND)
    include(${CGAL_USE_FILE})
else()
  message("ERROR: this program requires CGAL and will not be compiled.")
endif()



set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/../lib/Debug)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/../lib/Release) 

add_subdirectory(include)

