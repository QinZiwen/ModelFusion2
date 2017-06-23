cmake_minimum_required(VERSION 2.8)
project(ModelFusion2)

set( CMAKE_CXX_COMPILER "g++" )
set( CMAKE_BUILD_TYPE "Debug" ) # Release
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -O3 -pthread" )

set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )
link_directories( ${PROJECT_SOURCE_DIR}/lib "/usr/local/lib")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")
include(cmake/Utils.cmake)
include(cmake/Targets.cmake)
include(cmake/find_cuda.cmake)

find_package(OpenCV REQUIRED)
find_package(CUDA 8.0 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenNI REQUIRED)
find_package(PCL 1.8 REQUIRED)

add_definitions( ${PCL_DEFINITIONS} )
# link_libraries( ${PCL_LIBRARY_DIRS} )
# message("PCL include dir = " ${PCL_INCLUDE_DIRS})
# message("PCL libraries = " ${PCL_LIBRARIES})

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${OpenCV_INCLUDE_DIRS}
  ${OPENNI_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  #"/usr/local/include/pcl-1.8"
  ${CUDA_INCLUDE_DIRS}
  # ${EIGEN3_INCLUDE_DIR}           # #include <eigen3/Eigen/Core>
  "/usr/include/eigen3/"            # #include <Eigen/Core>
)

if(UNIX OR APPLE)
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler;-fPIC;")
endif()

list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_61,code=sm_61")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--disable-warnings;--ptxas-options=-v;-use_fast_math;-lineinfo;-std=c++11)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")

FILE(GLOB incs include/pcl/gpu/kinfu_large_scale/*.h*)
FILE(GLOB impl_incs include/pcl/gpu/kinfu_large_scale/impl/*.h*)
FILE(GLOB srcs src/*.cpp src/*.h*)
FILE(GLOB cuda src/cuda/*.cu src/cuda/*.h*)

CUDA_COMPILE(cuda_objs ${cuda})

add_library(pcl_gpu_kinfu_large_scale SHARED ${srcs} ${incs} ${impl_incs} ${cuda} ${cuda_objs})
target_link_libraries(pcl_gpu_kinfu_large_scale
  pcl_common
  pcl_io
  pcl_gpu_utils
  pcl_gpu_containers
  pcl_gpu_octree
  pcl_octree
  pcl_filters
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  ${OpenCV_LIBS}
  ${OPENNI_LIBRARY}
  ${PCL_LIBRARIES}
)

add_subdirectory(tools)
add_subdirectory(apps)