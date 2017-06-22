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

# set(SUBSYS_NAME gpu_kinfu_large_scale)
# set(SUBSYS_PATH gpu/kinfu_large_scale)
# set(SUBSYS_DESC "Kinect Fusion implementation, with volume shifting")
# set(SUBSYS_DEPS common visualization io gpu_containers gpu_utils geometry search octree filters kdtree features surface)
# 
# set(build FALSE)
# 
# find_package(OpenCV QUIET)
# 
# # OpenNI found?
# if(NOT WITH_OPENNI)
#     set(DEFAULT FALSE)
#     set(REASON "OpenNI was not found or was disabled by the user.")
# else()
#     set(DEFAULT TRUE)
#     set(REASON)
# endif()
# 
# PCL_SUBSYS_OPTION(build "${SUBSYS_NAME}" "${SUBSYS_DESC}" ${DEFAULT} "${REASON}")
# PCL_SUBSYS_DEPEND(build "${SUBSYS_NAME}" DEPS ${SUBSYS_DEPS})
# PCL_SET_SUBSYS_INCLUDE_DIR("${SUBSYS_NAME}" "${SUBSYS_PATH}")
# mark_as_advanced("BUILD_${SUBSYS_NAME}")
# 
# if (build)
# 	REMOVE_VTK_DEFINITIONS()
# 	FILE(GLOB incs include/pcl/gpu/kinfu_large_scale/*.h*)
# 	FILE(GLOB impl_incs include/pcl/gpu/kinfu_large_scale/impl/*.h*)
# 	FILE(GLOB srcs src/*.cpp src/*.h*)
# 	FILE(GLOB cuda src/cuda/*.cu src/cuda/*.h*)
# 	#FILE(GLOB tsdf src/cuda/tsdf_volume.cu)
# 	#FILE(GLOB est src/cuda/estimate_tranform.cu)
# 			
# 	source_group("Source Files\\cuda" FILES ${cuda} )    
# 	source_group("Source Files" FILES ${srcs} )		
# 	
# 	set(LIB_NAME "pcl_${SUBSYS_NAME}")
# 	include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include" 
# 	    "${CMAKE_CURRENT_SOURCE_DIR}/src"
# 	    ${CUDA_INCLUDE_DIRS}
# 	)
# 
# 	if (UNIX OR APPLE)
# 		set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;")			
# 	endif()
# 	
# 	if(NOT UNIX OR APPLE)
# 		add_definitions(-DPCLAPI_EXPORTS)
# 	endif()
# 	
# 	#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode;arch=compute_11,code=compute_11;-gencode;arch=compute_12,code=compute_12")
# 	
# 	if(TRUE)
# 		#list(REMOVE_ITEM cuda ${est})
# 		#CUDA_COMPILE(est_objs ${est})		
# 				   
# 		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")		
# 		CUDA_COMPILE(cuda_objs ${cuda})		
# 		
# 		#LIST(APPEND cuda ${est})
# 		#LIST(APPEND cuda_objs ${est_objs})
# 		
# 	else()		
# 		list(REMOVE_ITEM cuda ${tsdf})
# 		CUDA_COMPILE(cuda_objs ${cuda})		
# 		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "--ftz=true;--prec-div=false;--prec-sqrt=false") 
# 		CUDA_COMPILE(tsdf_obj ${tsdf})
# 	
# 		LIST(APPEND cuda ${tsdf})
# 		LIST(APPEND cuda_objs ${tsdf_obj})
# 	endif()
# 	 
# 	PCL_ADD_LIBRARY("${LIB_NAME}" "${SUBSYS_NAME}" ${srcs} ${incs} ${impl_incs} ${cuda} ${cuda_objs})
# 	target_link_libraries("${LIB_NAME}" pcl_common pcl_io pcl_gpu_utils pcl_gpu_containers pcl_gpu_octree pcl_octree pcl_filters "${OpenCV_LIBS}")
# 			
# 	set(EXT_DEPS "")
# 	#set(EXT_DEPS CUDA)
# 	PCL_MAKE_PKGCONFIG("${LIB_NAME}" "${SUBSYS_NAME}" "${SUBSYS_DESC}" "${SUBSYS_DEPS}" "${EXT_DEPS}" "" "" "")
# 
# 	# Install include files
# 	PCL_ADD_INCLUDES("${SUBSYS_NAME}" "${SUBSYS_PATH}" ${incs})
# 	
# 	add_subdirectory(keyModel)
# 	add_subdirectory(tools)
# endif()