./bin/orbslam2TSDF ~/Project/ORB_SLAM2/Vocabulary/ORBvoc.txt ~/Project/ORB_SLAM2/Examples/RGB-D/TUM1.yaml ~/Database/rgbd_dataset_freiburg1_room/ ~/Database/rgbd_dataset_freiburg1_room/associated.txt
#./bin/orbslam2TSDF ~/Project/ORB_SLAM2/Vocabulary/ORBvoc.txt ~/Project/ORB_SLAM2/Examples/RGB-D/TUM1.yaml ~/Database/rgbd_dataset_freiburg1_desk/ ~/Database/rgbd_dataset_freiburg1_desk/associated.txt

cd orbslam2_world
rm *.ply
../../pcl/build/bin/pcl_kinfu_largeScale_mesh_output ../orbslam2TSDF_world.pcd
cd ..
