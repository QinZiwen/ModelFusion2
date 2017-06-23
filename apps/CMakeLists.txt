
FILE(GLOB hdrs "*.h*")

## OFFLINE KINECT FUSION
set(the_target orbslam2TSDF)
set(srcs orbslam2TSDF.cpp ../tools/evaluation.cpp)

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
  libORB_SLAM2.so
)