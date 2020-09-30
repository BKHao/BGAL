cmake_minimum_required(VERSION 3.17)

file(GLOB_RECURSE cpp_files *.cpp) 

add_library(BKHao_CODE ${cpp_files})
target_link_libraries(BKHao_CODE ${Boost_LIBRARIES})
