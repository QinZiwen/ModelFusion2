
FILE(GLOB hdrs "*.h*")

## KINECT FUSION
set(the_target pcl_inline_kinfu_largeScale)
set(srcs inLine_kinfu.cpp evaluation.cpp)

add_executable(${the_target} ${srcs} ${hdrs})
target_link_libraries(${the_target}
  pcl_common
  pcl_io
  ${OPENNI_LIBRARIES}
  pcl_visualization
  pcl_gpu_kinfu_large_scale
  pcl_octree
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
)

## OFFLINE KINECT FUSION
set(the_target pcl_offline_kinfu_largeScale)
set(srcs offLine_kinfu.cpp evaluation.cpp)

add_executable(${the_target} ${srcs} ${hdrs})
target_link_libraries(${the_target}
  pcl_common
  pcl_io
  ${OPENNI_LIBRARIES}
  pcl_visualization
  pcl_gpu_kinfu_large_scale
  pcl_octree
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
)

## test openni2 grabber
set(the_target test_openni2_grabber)
set(srcs test_openni2_grabber.cpp)

add_executable(${the_target} ${srcs} ${hdrs})
target_link_libraries(${the_target}
  pcl_common
  pcl_io
  ${OPENNI_LIBRARIES}
  pcl_visualization
  pcl_gpu_kinfu_large_scale
  pcl_octree
  ${OpenCV_LIBS}
)