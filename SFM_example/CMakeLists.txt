cmake_minimum_required( VERSION 2.8 )
project( g2o_ba_example )

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -Wall -O2 -march=native" )


list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
list( APPEND CMAKE_MODULE_PATH /Users/patrickji/g2o/cmake_modules )

find_package( G2O REQUIRED )
find_package( Eigen3 REQUIRED )
find_package( cholmod)


include_directories( ${EIGEN3_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR} )
include_directories( /Users/patrickji/g2o)
include_directories( /Users/patrickji/g2o/build)

add_executable( ba_example ba_g2o.cpp )
target_link_libraries( ba_example 
    g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension g2o_types_sba
    ${CHOLMOD_LIBRARIES}
    )
