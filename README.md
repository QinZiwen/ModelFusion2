ModelFusion2
============

# Introduction
Getting dense reconstruction based on kinfu_large_scale and orbslam2.

- kinfu_large_scale: [https://github.com/PointCloudLibrary/pcl/tree/master/gpu/kinfu_large_scale](https://github.com/PointCloudLibrary/pcl/tree/master/gpu/kinfu_large_scale)
- orbslam2: [https://github.com/raulmur/ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2)

# Key changes/features

- dense reconstruction
- real-time
- robust tracking

# Dependencies

- pcl 1.8
- OpenCV
- cuda
- Eigen3

# Compile
```
$ git clone https://github.com/QinZiwen/ModelFusion2.git
$ cd ModelFusion2
$ mkdir build; cd build; make; cd ..
```

# Usage
- online fusion by kinect2
```
$ ./bin/pcl_inline_kinfu_largeScale
```

- offline fusion with TUM Database
```
$ ./bin/pcl_offline_kinfu_largeScale -eval ~/Database/rgbd_dataset_freiburg1_room/
```