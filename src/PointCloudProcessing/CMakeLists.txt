cmake_minimum_required(VERSION 3.17)

set(BGAL_Registration_SRC        
        Registration/ICP/ICP.cpp
        )

# Get static lib
add_library(Registration STATIC ${BGAL_Registration_SRC})
target_link_libraries(Registration Algorithm BaseShape ${Boost_LIBRARIES})
set_target_properties(Registration PROPERTIES VERSION ${VERSION})
set_target_properties(Registration PROPERTIES CLEAN_DIRECT_OUTPUT 1)


target_include_directories(Registration PUBLIC
	$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>)
