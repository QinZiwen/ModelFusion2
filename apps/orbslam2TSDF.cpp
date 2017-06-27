#define _CRT_SECURE_NO_DEPRECATE

#include <iostream>

#include <pcl/console/parse.h>

#include <boost/filesystem.hpp>
#include <boost/graph/graph_concepts.hpp>
#include <boost/shared_ptr.hpp>

#include <pcl/gpu/kinfu_large_scale/kinfu.h>
#include <pcl/gpu/kinfu_large_scale/raycaster.h>
#include <pcl/gpu/kinfu_large_scale/marching_cubes.h>
#include <pcl/gpu/kinfu_large_scale/screenshot_manager.h>
#include <pcl/gpu/containers/initialization.h>

#include <pcl/common/time.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/io/openni2_grabber.h>
#include <pcl/io/oni_grabber.h>
#include <pcl/io/pcd_grabber.h>
#include <pcl/common/angles.h>

#ifdef HAVE_OPENCV  
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

#include "tools/color_handler.h"
#include "tools/evaluation.h"
#include "orbslam2/System.h"

typedef pcl::ScopeTime ScopeTimeT;

using namespace std;
using namespace pcl;
using namespace Eigen;

using namespace pcl::gpu::kinfuLS;

using pcl::gpu::DeviceArray;
using pcl::gpu::DeviceArray2D;
using pcl::gpu::PtrStepSz;

namespace pc = pcl::console;

namespace pcl
{
  namespace gpu
  {
    namespace kinfuLS
    {
      void paint3DView (const KinfuTracker::View& rgb24, KinfuTracker::View& view, float colors_weight = 0.5f);
      void mergePointNormal (const DeviceArray<PointXYZ>& cloud, const DeviceArray<Normal>& normals, DeviceArray<PointNormal>& output);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

vector<string> getPcdFilesInDir(const string& directory)
{
  namespace fs = boost::filesystem;
  fs::path dir(directory);

  std::cout << "path: " << directory << std::endl;
  if (directory.empty() || !fs::exists(dir) || !fs::is_directory(dir))
          PCL_THROW_EXCEPTION (pcl::IOException, "No valid PCD directory given!\n");

  vector<string> result;
  fs::directory_iterator pos(dir);
  fs::directory_iterator end;           

  for(; pos != end ; ++pos)
    if (fs::is_regular_file(pos->status()) )
      if (fs::extension(*pos) == ".pcd")
      {
#if BOOST_FILESYSTEM_VERSION == 3
        result.push_back (pos->path ().string ());
#else
        result.push_back (pos->path ());
#endif
        cout << "added: " << result.back() << endl;
      }

  return result;  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SampledScopeTime : public StopWatch
{          
  enum { EACH = 33 };
  SampledScopeTime(int& time_ms) : time_ms_(time_ms) {}
  ~SampledScopeTime()
  {
    static int i_ = 0;
    static boost::posix_time::ptime starttime_ = boost::posix_time::microsec_clock::local_time();
    time_ms_ += getTime ();
    if (i_ % EACH == 0 && i_)
    {
      boost::posix_time::ptime endtime_ = boost::posix_time::microsec_clock::local_time();
      cout << "Average frame time = " << time_ms_ / EACH << "ms ( " << 1000.f * EACH / time_ms_ << "fps )"
           << "( real: " << 1000.f * EACH / (endtime_-starttime_).total_milliseconds() << "fps )"  << endl;
      time_ms_ = 0;
      starttime_ = endtime_;
    }
    ++i_;
  }
private:    
  int& time_ms_;    
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
setViewerPose (visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose)
{
  Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f (0, 0, 0);
  Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f (0, 0, 1) + pos_vector;
  Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f (0, -1, 0);
  viewer.setCameraPosition (pos_vector[0], pos_vector[1], pos_vector[2],
                            look_at_vector[0], look_at_vector[1], look_at_vector[2],
                            up_vector[0], up_vector[1], up_vector[2]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Eigen::Affine3f 
getViewerPose (visualization::PCLVisualizer& viewer)
{
  Eigen::Affine3f pose = viewer.getViewerPose();
  Eigen::Matrix3f rotation = pose.linear();

  Matrix3f axis_reorder;  
  axis_reorder << 0,  0,  1,
          -1,  0,  0,
          0, -1,  0;

  rotation = rotation * axis_reorder;
  pose.linear() = rotation;
  return pose;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename CloudT> void
writeCloudFile (int format, const CloudT& cloud);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
writePolygonMeshFile (int format, const pcl::PolygonMesh& mesh);


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename MergedT, typename PointT>
typename PointCloud<MergedT>::Ptr merge(const PointCloud<PointT>& points, const PointCloud<RGB>& colors)
{    
  typename PointCloud<MergedT>::Ptr merged_ptr(new PointCloud<MergedT>());

  pcl::copyPointCloud (points, *merged_ptr);      
  for (size_t i = 0; i < colors.size (); ++i)
    merged_ptr->points[i].rgba = colors.points[i].rgba;

  return merged_ptr;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

boost::shared_ptr<pcl::PolygonMesh> convertToMesh(const DeviceArray<PointXYZ>& triangles)
{ 
  if (triangles.empty())
          return boost::shared_ptr<pcl::PolygonMesh>();

  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.width  = (int)triangles.size();
  cloud.height = 1;
  triangles.download(cloud.points);

  boost::shared_ptr<pcl::PolygonMesh> mesh_ptr( new pcl::PolygonMesh() ); 
  pcl::toPCLPointCloud2(cloud, mesh_ptr->cloud);

  mesh_ptr->polygons.resize (triangles.size() / 3);
  for (size_t i = 0; i < mesh_ptr->polygons.size (); ++i)
  {
    pcl::Vertices v;
    v.vertices.push_back(i*3+0);
    v.vertices.push_back(i*3+2);
    v.vertices.push_back(i*3+1);              
    mesh_ptr->polygons[i] = v;
  }    
  return mesh_ptr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct CurrentFrameCloudView
{
  CurrentFrameCloudView() : cloud_device_ (480, 640), cloud_viewer_ ("Frame Cloud Viewer")
  {
    cloud_ptr_ = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);

    cloud_viewer_.setBackgroundColor (0, 0, 0.15);
    cloud_viewer_.addCoordinateSystem (1.0, "global");
    cloud_viewer_.initCameraParameters ();
    cloud_viewer_.setPosition (0, 500);
    cloud_viewer_.setSize (640, 480);
    cloud_viewer_.setCameraClipDistances (0.01, 10.01);
  }

  void
  show (const KinfuTracker& kinfu)
  {
    kinfu.getLastFrameCloud (cloud_device_);

    int c;
    cloud_device_.download (cloud_ptr_->points, c);
    cloud_ptr_->width = cloud_device_.cols ();
    cloud_ptr_->height = cloud_device_.rows ();
    cloud_ptr_->is_dense = false;

    cloud_viewer_.removeAllPointClouds ();
    cloud_viewer_.addPointCloud<PointXYZ>(cloud_ptr_);
    cloud_viewer_.spinOnce ();
  }

  void
  setViewerPose (const Eigen::Affine3f& viewer_pose) {
    ::setViewerPose (cloud_viewer_, viewer_pose);
  }

  PointCloud<PointXYZ>::Ptr cloud_ptr_;
  DeviceArray2D<PointXYZ> cloud_device_;
  visualization::PCLVisualizer cloud_viewer_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct ImageView
{
  ImageView() : paint_image_ (false), accumulate_views_ (false)
  {
    viewerScene_.setWindowTitle ("View3D from ray tracing");
    viewerScene_.setPosition (0, 0);
    viewerDepth_.setWindowTitle ("Kinect Depth stream");
    viewerDepth_.setPosition (640, 0);
    viewerColor_.setWindowTitle ("Kinect RGB stream");
  }

  void
  showScene (KinfuTracker& kinfu, const PtrStepSz<const pcl::gpu::kinfuLS::PixelRGB>& rgb24, bool haveImage, Eigen::Affine3f* pose_ptr = 0)
  {
    if (pose_ptr)
    {
      raycaster_ptr_->run ( kinfu.volume (), *pose_ptr, kinfu.getCyclicalBufferStructure () ); //says in cmake it does not know it
      raycaster_ptr_->generateSceneView(view_device_);
    }
    else
    {
      kinfu.getImage (view_device_);
    }

    if (haveImage && !pose_ptr)
    {
      colors_device_.upload (rgb24.data, rgb24.step * 3, rgb24.rows, rgb24.cols);
      paint3DView (colors_device_, view_device_);
    }

    int cols;
    view_device_.download (view_host_, cols);
    viewerScene_.showRGBImage (reinterpret_cast<unsigned char*> (&view_host_[0]), view_device_.cols (), view_device_.rows ());    
  }
  
  void
  showScene (KinfuTracker& kinfu, bool registration, Eigen::Affine3f* pose_ptr = 0)
  {
    if (pose_ptr)
    {
      raycaster_ptr_->run ( kinfu.volume (), *pose_ptr, kinfu.getCyclicalBufferStructure () ); //says in cmake it does not know it
      raycaster_ptr_->generateSceneView(view_device_);
    }
    else
    {
      kinfu.getImage (view_device_);
    }

    int cols;
    view_device_.download (view_host_, cols);
    viewerScene_.showRGBImage (reinterpret_cast<unsigned char*> (&view_host_[0]), view_device_.cols (), view_device_.rows ());    
  }

  void
  showDepth (const PtrStepSz<const unsigned short>& depth) 
  { 
    viewerDepth_.showShortImage (depth.data, depth.cols, depth.rows, 0, 5000, true); 
  }
  
	void showRGB(const PtrStepSz<const pcl::gpu::kinfuLS::PixelRGB>& rgb24)
	{
		viewerColor_.showRGBImage ((unsigned char*)rgb24.data, rgb24.cols, rgb24.rows);
	}

  void
  showGeneratedDepth (KinfuTracker& kinfu, const Eigen::Affine3f& pose)
  {            
    raycaster_ptr_->run(kinfu.volume(), pose, kinfu.getCyclicalBufferStructure ());
    raycaster_ptr_->generateDepthImage(generated_depth_);    

    int c;
    vector<unsigned short> data;
    generated_depth_.download(data, c);

    viewerDepth_.showShortImage (&data[0], generated_depth_.cols(), generated_depth_.rows(), 0, 5000, true);
  }

  void
  toggleImagePaint()
  {
    paint_image_ = !paint_image_;
    cout << "Paint image: " << (paint_image_ ? "On   (requires registration mode)" : "Off") << endl;
  }

  bool paint_image_;
  bool accumulate_views_;

  visualization::ImageViewer viewerScene_; //view the raycasted model
  visualization::ImageViewer viewerDepth_; //view the current depth map
  visualization::ImageViewer viewerColor_;

  KinfuTracker::View view_device_;
  KinfuTracker::View colors_device_;
  vector<pcl::gpu::kinfuLS::PixelRGB> view_host_;

  RayCaster::Ptr raycaster_ptr_;

  KinfuTracker::DepthMap generated_depth_;

#ifdef HAVE_OPENCV
  vector<cv::Mat> views_;
#endif
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// View the volume as 3D points
struct SceneCloudView
{
  enum { GPU_Connected6 = 0, CPU_Connected6 = 1, CPU_Connected26 = 2 };

  SceneCloudView() : extraction_mode_ (GPU_Connected6), compute_normals_ (false), valid_combined_ (false), cube_added_(false), cloud_viewer_ ("Scene Cloud Viewer")
  {
    cloud_ptr_ = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);
    normals_ptr_ = PointCloud<Normal>::Ptr (new PointCloud<Normal>);
    combined_ptr_ = PointCloud<PointNormal>::Ptr (new PointCloud<PointNormal>);
    point_colors_ptr_ = PointCloud<RGB>::Ptr (new PointCloud<RGB>);

    cloud_viewer_.setBackgroundColor (0, 0, 0);
    cloud_viewer_.addCoordinateSystem (1.0, "global");
    cloud_viewer_.initCameraParameters ();
    cloud_viewer_.setPosition (0, 500);
    cloud_viewer_.setSize (640, 480);
    cloud_viewer_.setCameraClipDistances (0.01, 10.01);

    cloud_viewer_.addText ("H: print help", 2, 15, 20, 34, 135, 246);         
    cloud_viewer_.addText ("ICP State: ", 450, 55, 20, 0.0, 1.0, 0.0, "icp");
    cloud_viewer_.addText ("Press 'S' to save the current world", 450, 35, 10, 0.0, 1.0, 0.0, "icp_save");
    cloud_viewer_.addText ("Press 'R' to reset the system", 450, 15, 10, 0.0, 1.0, 0.0, "icp_reset");
  }
  
  inline void 
  drawCamera (Eigen::Affine3f& pose, const string& name, double r, double g, double b)
  {
    double focal = 575;
    double height = 480;
    double width = 640;
    
    // create a 5-point visual for each camera
    pcl::PointXYZ p1, p2, p3, p4, p5;
    p1.x=0; p1.y=0; p1.z=0;
    double angleX = RAD2DEG (2.0 * atan (width / (2.0*focal)));
    double angleY = RAD2DEG (2.0 * atan (height / (2.0*focal)));
    double dist = 0.75;
    double minX, minY, maxX, maxY;
    maxX = dist*tan (atan (width / (2.0*focal)));
    minX = -maxX;
    maxY = dist*tan (atan (height / (2.0*focal)));
    minY = -maxY;
    p2.x=minX; p2.y=minY; p2.z=dist;
    p3.x=maxX; p3.y=minY; p3.z=dist;
    p4.x=maxX; p4.y=maxY; p4.z=dist;
    p5.x=minX; p5.y=maxY; p5.z=dist;
    p1=pcl::transformPoint (p1, pose);
    p2=pcl::transformPoint (p2, pose);
    p3=pcl::transformPoint (p3, pose);
    p4=pcl::transformPoint (p4, pose);
    p5=pcl::transformPoint (p5, pose);
    std::stringstream ss;
    ss.str ("");
    ss << name << "_line1";
    cloud_viewer_.addLine (p1, p2, r, g, b, ss.str ());
    ss.str ("");
    ss << name << "_line2";
    cloud_viewer_.addLine (p1, p3, r, g, b, ss.str ());
    ss.str ("");
    ss << name << "_line3";
    cloud_viewer_.addLine (p1, p4, r, g, b, ss.str ());
    ss.str ("");
    ss << name << "_line4";
    cloud_viewer_.addLine (p1, p5, r, g, b, ss.str ());
    ss.str ("");
    ss << name << "_line5";
    cloud_viewer_.addLine (p2, p5, r, g, b, ss.str ());
    ss.str ("");
    ss << name << "_line6";
    cloud_viewer_.addLine (p5, p4, r, g, b, ss.str ());
    ss.str ("");
    ss << name << "_line7";
    cloud_viewer_.addLine (p4, p3, r, g, b, ss.str ());
    ss.str ("");
    ss << name << "_line8";
    cloud_viewer_.addLine (p3, p2, r, g, b, ss.str ());    
  }
  
  inline void 
  removeCamera (const string& name)
  {
    cloud_viewer_.removeShape (name);
    std::stringstream ss;
    ss.str ("");
    ss << name << "_line1";
    cloud_viewer_.removeShape (ss.str ());
    ss.str ("");
    ss << name << "_line2";
    cloud_viewer_.removeShape (ss.str ());
    ss.str ("");
    ss << name << "_line3";
    cloud_viewer_.removeShape (ss.str ());
    ss.str ("");
    ss << name << "_line4";
    cloud_viewer_.removeShape (ss.str ());
    ss.str ("");
    ss << name << "_line5";
    cloud_viewer_.removeShape (ss.str ());
    ss.str ("");
    ss << name << "_line6";
    cloud_viewer_.removeShape (ss.str ());
    ss.str ("");
    ss << name << "_line7";
    cloud_viewer_.removeShape (ss.str ());
    ss.str ("");
    ss << name << "_line8";
    cloud_viewer_.removeShape (ss.str ());
  }

  void
  displayICPState (KinfuTracker& kinfu, bool was_lost_)
  {
    string name = "last_good_track";
    string name_estimate = "last_good_estimate";
    if (was_lost_ && !kinfu.icpIsLost ()) //execute only when ICP just recovered (i.e. was_lost_ == true && icpIsLost == false)
    {
      removeCamera(name);
      removeCamera(name_estimate);
      clearClouds(false);
      cloud_viewer_.updateText ("ICP State: OK", 450, 55, 20, 0.0, 1.0, 0.0, "icp");
      cloud_viewer_.updateText ("Press 'S' to save the current world", 450, 35, 10, 0.0, 1.0, 0.0, "icp_save");
      cloud_viewer_.updateText ("Press 'R' to reset the system", 450, 15, 10, 0.0, 1.0, 0.0, "icp_reset");
    }
    else if (!was_lost_ && kinfu.icpIsLost()) //execute only when we just lost ourselves (i.e. was_lost_ = false && icpIsLost == true)
    { 
      // draw position of the last good track
      Eigen::Affine3f last_pose = kinfu.getCameraPose();
      drawCamera(last_pose, name, 0.0, 1.0, 0.0);
      cloud_viewer_.updateText ("ICP State: LOST", 450, 55, 20, 1.0, 0.0, 0.0, "icp");
      cloud_viewer_.updateText ("Press 'S' to save the current world", 450, 35, 10, 1.0, 0.0, 0.0, "icp_save");
      cloud_viewer_.updateText ("Press 'R' to reset the system", 450, 15, 10, 1.0, 0.0, 0.0, "icp_reset");
    }    
    
    if( kinfu.icpIsLost() )
    {
      removeCamera(name_estimate);
       // draw current camera estimate
      Eigen::Affine3f last_pose_estimate = kinfu.getLastEstimatedPose();
      drawCamera(last_pose_estimate, name_estimate, 1.0, 0.0, 0.0);      
    }
//       cout << "current estimated pose: " << kinfu.getLastEstimatedPose().translation() << std::endl << kinfu.getLastEstimatedPose().linear() << std::endl;
//     
  }
  
  void
  show (KinfuTracker& kinfu, bool integrate_colors)
  {
    viewer_pose_ = kinfu.getCameraPose();

    ScopeTimeT time ("PointCloud Extraction");
    cout << "\nGetting cloud... " << flush;

    valid_combined_ = false;

    if (extraction_mode_ != GPU_Connected6)     // So use CPU
    {
      kinfu.volume().fetchCloudHost (*cloud_ptr_, extraction_mode_ == CPU_Connected26);
    }
    else
    {
      DeviceArray<PointXYZ> extracted = kinfu.volume().fetchCloud (cloud_buffer_device_);             

      if (compute_normals_)
      {
        kinfu.volume().fetchNormals (extracted, normals_device_);
        pcl::gpu::kinfuLS::mergePointNormal (extracted, normals_device_, combined_device_);
        combined_device_.download (combined_ptr_->points);
        combined_ptr_->width = (int)combined_ptr_->points.size ();
        combined_ptr_->height = 1;

        valid_combined_ = true;
      }
      else
      {
        extracted.download (cloud_ptr_->points);
        cloud_ptr_->width = (int)cloud_ptr_->points.size ();
        cloud_ptr_->height = 1;
      }

      if (integrate_colors)
      {
        kinfu.colorVolume().fetchColors(extracted, point_colors_device_);
        point_colors_device_.download(point_colors_ptr_->points);
        point_colors_ptr_->width = (int)point_colors_ptr_->points.size ();
        point_colors_ptr_->height = 1;
      }
      else
        point_colors_ptr_->points.clear();
    }
    size_t points_size = valid_combined_ ? combined_ptr_->points.size () : cloud_ptr_->points.size ();
    cout << "Done.  Cloud size: " << points_size / 1000 << "K" << endl;

    cloud_viewer_.removeAllPointClouds ();    
    if (valid_combined_)
    {
      visualization::PointCloudColorHandlerRGBHack<PointNormal> rgb(combined_ptr_, point_colors_ptr_);
      cloud_viewer_.addPointCloud<PointNormal> (combined_ptr_, rgb, "Cloud");
      cloud_viewer_.addPointCloudNormals<PointNormal>(combined_ptr_, 50);
    }
    else
    {
      visualization::PointCloudColorHandlerRGBHack<PointXYZ> rgb(cloud_ptr_, point_colors_ptr_);
      cloud_viewer_.addPointCloud<PointXYZ> (cloud_ptr_, rgb);
    }    
  }

  void
  toggleCube(const Eigen::Vector3f& size)
  {
    if (cube_added_)
      cloud_viewer_.removeShape("cube");
    else
      cloud_viewer_.addCube(size*0.5, Eigen::Quaternionf::Identity(), size(0), size(1), size(2));

    cube_added_ = !cube_added_;
  }

  void
  toggleExtractionMode ()
  {
    extraction_mode_ = (extraction_mode_ + 1) % 3;
    switch (extraction_mode_)
    {
      case 0: cout << "Cloud extraction mode: GPU, Connected-6" << endl; break;
      case 1: cout << "Cloud extraction mode: CPU, Connected-6    (requires a lot of memory)" << endl; break;
      case 2: cout << "Cloud extraction mode: CPU, Connected-26   (requires a lot of memory)" << endl; break;
    }         
  }

  void
  toggleNormals ()
  {
    compute_normals_ = !compute_normals_;
    cout << "Compute normals: " << (compute_normals_ ? "On" : "Off") << endl;
  }

  void
  clearClouds (bool print_message = false)
  {
    cloud_viewer_.removeAllPointClouds ();
    cloud_ptr_->points.clear ();
    normals_ptr_->points.clear ();    
    if (print_message)
      cout << "Clouds/Meshes were cleared" << endl;
  }

  void
  showMesh(KinfuTracker& kinfu, bool /*integrate_colors*/)
  {
    ScopeTimeT time ("Mesh Extraction");
    cout << "\nGetting mesh... " << flush;

    if (!marching_cubes_)
      marching_cubes_ = MarchingCubes::Ptr( new MarchingCubes() );

    DeviceArray<PointXYZ> triangles_device = marching_cubes_->run(kinfu.volume(), triangles_buffer_device_);    
    mesh_ptr_ = convertToMesh(triangles_device);

    cloud_viewer_.removeAllPointClouds ();
    if (mesh_ptr_)
      cloud_viewer_.addPolygonMesh(*mesh_ptr_);	

    cout << "Done.  Triangles number: " << triangles_device.size() / MarchingCubes::POINTS_PER_TRIANGLE / 1000 << "K" << endl;
  }

  int extraction_mode_;
  bool compute_normals_;
  bool valid_combined_;
  bool cube_added_;

  Eigen::Affine3f viewer_pose_;

  visualization::PCLVisualizer cloud_viewer_;

  PointCloud<PointXYZ>::Ptr cloud_ptr_;
  PointCloud<Normal>::Ptr normals_ptr_;

  DeviceArray<PointXYZ> cloud_buffer_device_;
  DeviceArray<Normal> normals_device_;

  PointCloud<PointNormal>::Ptr combined_ptr_;
  DeviceArray<PointNormal> combined_device_;  

  DeviceArray<RGB> point_colors_device_; 
  PointCloud<RGB>::Ptr point_colors_ptr_;

  MarchingCubes::Ptr marching_cubes_;
  DeviceArray<PointXYZ> triangles_buffer_device_;

  boost::shared_ptr<pcl::PolygonMesh> mesh_ptr_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct Impl
{
   cv::Mat depth_buffer;
   cv::Mat rgb_buffer;
};

struct KinFuLSApp
{
  enum { PCD_BIN = 1, PCD_ASCII = 2, PLY = 3, MESH_PLY = 7, MESH_VTK = 8 };

  KinFuLSApp(float vsz, float shiftDistance, int snapshotRate, const string &strVocFile, const string &strSettingsFile) : exit_ (false), scan_ (false), scan_mesh_(false), scan_volume_ (false), independent_camera_ (false),
          registration_ (false), integrate_colors_ (false), pcd_source_ (false), focal_length_(-1.f), was_lost_(false), time_ms_ (0)
  {    
    //Init Kinfu Tracker
    Eigen::Vector3f volume_size = Vector3f::Constant (vsz/*meters*/);    

    PCL_WARN ("--- CURRENT SETTINGS ---\n");
    PCL_INFO ("Volume size is set to %.2f meters\n", vsz);
    PCL_INFO ("Volume will shift when the camera target point is farther than %.2f meters from the volume center\n", shiftDistance);
    PCL_INFO ("The target point is located at [0, 0, %.2f] in camera coordinates\n", 0.6*vsz);
    PCL_WARN ("------------------------\n");

    // warning message if shifting distance is abnormally big compared to volume size
    if(shiftDistance > 2.5 * vsz)
      PCL_WARN ("WARNING Shifting distance (%.2f) is very large compared to the volume size (%.2f).\nYou can modify it using --shifting_distance.\n", shiftDistance, vsz);

    kinfu_ = new pcl::gpu::kinfuLS::KinfuTracker(volume_size, shiftDistance);

    Eigen::Matrix3f R = Eigen::Matrix3f::Identity ();   // * AngleAxisf( pcl::deg2rad(-30.f), Vector3f::UnitX());
    //Eigen::Vector3f t = volume_size * 0.5f - Vector3f (0, 0, volume_size (2) / 2 * 1.2f);
	Eigen::Vector3f t(0,0,0);
    
    Eigen::Affine3f pose = Eigen::Translation3f (t) * Eigen::AngleAxisf (R);

    kinfu_->setInitialCameraPose (pose);
    kinfu_->volume().setTsdfTruncDist (0.030f/*meters*/);
    kinfu_->setIcpCorespFilteringParams (0.1f/*meters*/, sin ( pcl::deg2rad(20.f) ));
    //kinfu_->setDepthTruncationForICP(3.f/*meters*/);
    kinfu_->setCameraMovementThreshold(0.001f);
	
	cv::FileStorage fSettings(strSettingsFile, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];
	kinfu_->setDepthIntrinsics(fx, fy, cx, cy);

    //Init KinFuLSApp            
    tsdf_cloud_ptr_ = pcl::PointCloud<pcl::PointXYZI>::Ptr (new pcl::PointCloud<pcl::PointXYZI>);
    image_view_.raycaster_ptr_ = RayCaster::Ptr( new RayCaster(kinfu_->rows (), kinfu_->cols ()) );

    scene_cloud_view_.cloud_viewer_.registerKeyboardCallback (keyboard_callback, (void*)this);
    image_view_.viewerScene_.registerKeyboardCallback (keyboard_callback, (void*)this);
    image_view_.viewerDepth_.registerKeyboardCallback (keyboard_callback, (void*)this);

    scene_cloud_view_.toggleCube(volume_size);
    frame_counter_ = 0;
	
	orbslam2_ = new ORB_SLAM2::System(strVocFile, strSettingsFile, ORB_SLAM2::System::RGBD);
	impl_.reset( new Impl() );
	
	PCL_WARN ("-- KinFuLSApp create nice!\n");
  }

  ~KinFuLSApp()
  {
    if (evaluation_ptr_)
      evaluation_ptr_->saveAllPoses(*kinfu_);
	
	if(orbslam2_)
	{
		orbslam2_->Shutdown();
		delete orbslam2_;
	}
  }

  void
  initCurrentFrameView ()
  {
    current_frame_cloud_view_ = boost::shared_ptr<CurrentFrameCloudView>(new CurrentFrameCloudView ());
    current_frame_cloud_view_->cloud_viewer_.registerKeyboardCallback (keyboard_callback, (void*)this);
    current_frame_cloud_view_->setViewerPose (kinfu_->getCameraPose ());
  }

  void 
  toggleColorIntegration()
  {
    if(registration_)
    {
      const int max_color_integration_weight = 2;
      kinfu_->initColorIntegration(max_color_integration_weight);
      integrate_colors_ = true;      
    }    
    cout << "Color integration: " << (integrate_colors_ ? "On" : "Off ( requires registration mode )") << endl;
  }
  
  void 
  toggleColorIntegration_database()
  {
	const int max_color_integration_weight = 2;
	kinfu_->initColorIntegration(max_color_integration_weight);
	integrate_colors_ = true;    
    cout << "Color integration: " << (integrate_colors_ ? "On" : "Off ( requires registration mode )") << endl;
  }

  void
  toggleIndependentCamera()
  {
    independent_camera_ = !independent_camera_;
    cout << "Camera mode: " << (independent_camera_ ?  "Independent" : "Bound to Kinect pose") << endl;
  }

  void
  toggleEvaluationMode(const string& eval_folder, const string& match_file = string())
  {
    evaluation_ptr_ = Evaluation::Ptr( new Evaluation(eval_folder) );
    if (!match_file.empty())
      evaluation_ptr_->setMatchFile(match_file);

    kinfu_->setDepthIntrinsics (evaluation_ptr_->fx, evaluation_ptr_->fy, evaluation_ptr_->cx, evaluation_ptr_->cy);
    image_view_.raycaster_ptr_ = RayCaster::Ptr( new RayCaster(kinfu_->rows (), kinfu_->cols (), evaluation_ptr_->fx, evaluation_ptr_->fy, evaluation_ptr_->cx, evaluation_ptr_->cy) );
	
	PCL_WARN("-- toggleEvaluationMode nice!\n");
  }
  
	bool executeUseDataBase(cv::Mat &depthMat, cv::Mat &imageMat,const PtrStepSz<const unsigned short>& depth, const PtrStepSz<const pcl::gpu::kinfuLS::PixelRGB>& rgb24, double tframe)
	{        
		bool has_image = false;
		frame_counter_++;
		cout << "frame_counter = " << frame_counter_ << endl;

		was_lost_ = kinfu_->icpIsLost();
		if(was_lost_)
		{
			PCL_WARN("kinfu has been lost\n");
			return false;
		}

		depth_device_.upload (depth.data, depth.step, depth.rows, depth.cols);

		if (integrate_colors_)
			image_view_.colors_device_.upload (rgb24.data, rgb24.step * 3, rgb24.rows, rgb24.cols);
		
		//cv::imshow("depth", depthMat);
		//cv::waitKey();
		
		cv::Mat T;
		{
			SampledScopeTime fps(time_ms_);

			//run kinfu algorithm
			//if (integrate_colors_)
			//	has_image = (*kinfu_) (depth_device_, image_view_.colors_device_);
			//else
			//	has_image = (*kinfu_) (depth_device_);
			T = orbslam2_->TrackRGBD(imageMat, depthMat, tframe);
			
			if(T.empty())
			{
				cout << "orbslam2 lost!" <<endl;
				return false;
			}
			else
				has_image = true;
			
			//cout << "T = " << endl << T << endl;
			
			cv::Mat Rcw = T.rowRange(0,3).colRange(0,3);
			cv::Mat tcw = T.rowRange(0,3).col(3);

			cv::Mat Rwc = Rcw.t();
			cv::Mat twc = -Rcw.t()*tcw;
			
			//cv::Mat &Rwc = Rcw;
			//cv::Mat &twc = tcw;
						
			KinfuTracker::Matrix3frm KRwc;
			KinfuTracker::Vector3f Ktwc;
			
			KRwc(0,0) = Rwc.at<float>(0,0); KRwc(0,1) = Rwc.at<float>(0,1); KRwc(0,2) = Rwc.at<float>(0,2);
			KRwc(1,0) = Rwc.at<float>(1,0); KRwc(1,1) = Rwc.at<float>(1,1); KRwc(1,2) = Rwc.at<float>(1,2);
			KRwc(2,0) = Rwc.at<float>(2,0); KRwc(2,1) = Rwc.at<float>(2,1); KRwc(2,2) = Rwc.at<float>(2,2);
			
			Ktwc(0) = twc.at<float>(0); Ktwc(1) = twc.at<float>(1); Ktwc(2) = twc.at<float>(2);
			
			kinfu_->keyModel_integrateWithPose(depth_device_, KRwc, Ktwc);
		}
		image_view_.showDepth (depth);
		image_view_.showRGB(rgb24);
		
		//image_view_.showGeneratedDepth(*kinfu_, kinfu_->getCameraPose());

// 		if (scan_mesh_)
// 		{
// 			scan_mesh_ = false;
// 			scene_cloud_view_.showMesh(*kinfu_, integrate_colors_);
// 		}

		if (has_image)
		{
			Eigen::Affine3f viewer_pose = kinfu_->getCameraPose ();
			image_view_.showScene (*kinfu_, rgb24, integrate_colors_, &viewer_pose);
		}    

// 		if (current_frame_cloud_view_)
// 			current_frame_cloud_view_->show (*kinfu_);

// 		// if ICP is lost, we show the world from a farther view point
// 		if(kinfu_->icpIsLost())
// 		{
// 			setViewerPose (scene_cloud_view_.cloud_viewer_, kinfu_->getCameraPose () * delta_lost_pose_);
// 		}
// 		else
// 		if (!independent_camera_)
// 			setViewerPose (scene_cloud_view_.cloud_viewer_, kinfu_->getCameraPose ());
// 
		// display ICP state
		scene_cloud_view_.displayICPState (*kinfu_, was_lost_);
		
		return true;
	}
  
	void startMainLoop_use_database (vector<double> vTimestamps, vector<string> vstrRGB, vector<string> vstrDepth)
	{
		PCL_WARN("-- RUNNING MAIN LOOP USING DATABASE ... \n");

		pcl::gpu::PtrStepSz<const unsigned short> depth;
		pcl::gpu::PtrStepSz<const pcl::gpu::kinfuLS::PixelRGB> rgb24;
		cv::Mat depthMat, imageMat;
		double tframe = 0;
		double stamp = 0;
		while (!exit_ && !scene_cloud_view_.cloud_viewer_.wasStopped () && !image_view_.viewerScene_.wasStopped () && !this->kinfu_->isFinished () && stamp < vTimestamps.size())
		{
			tframe = vTimestamps[stamp];
			
			if(!ImageGrab(vstrDepth[stamp], vstrRGB[stamp], depthMat, imageMat, depth, rgb24))
			{
				continue;
			}
			
			try 
			{
				if(! this->executeUseDataBase(depthMat, imageMat, depth, rgb24, tframe))
				{
					break;
				}
			}
			catch (const std::bad_alloc& /*e*/) { cout << "Bad alloc" << endl; break; }
			catch (const std::exception& /*e*/) { cout << "Exception" << endl; break; }

			scene_cloud_view_.cloud_viewer_.spinOnce (3);
			++stamp;                  
		} 

		exit_ = true;
		boost::this_thread::sleep (boost::posix_time::millisec (100));
		
		kinfu_->extractAndSaveWorld("orbslam2TSDF_world.pcd");
		orbslam2_->SaveTrajectoryTUM("orbslam2TSDF_pose.txt");
	}
	
	bool MatToDevice(const cv::Mat &depthMat, const cv::Mat &imageMat, PtrStepSz<const unsigned short>& depth, PtrStepSz<const pcl::gpu::kinfuLS::PixelRGB>& rgb24)
	{
		cv::Mat depth_buffer;
		cv::Mat rgb_buffer;
		if(depthMat.empty())
		{
			std::cout << "depth_file empty!" << std::endl;
			return false;
		}

		if (depthMat.elemSize() != sizeof(unsigned short))
		{
			cout << "Image was not opend in 16-bit format. Please use OpenCV 2.3.1 or higher" << endl;
			exit(1);
		}

		// Datasets are with factor 5000 (pixel to m) 
		// http://cvpr.in.tum.de/data/datasets/rgbd-dataset/file_formats#color_images_and_depth_maps

		depthMat.convertTo(depth_buffer, depthMat.type(), 0.2);   //使用tum数据库的时候，这样可以是scale统一为1000.（tum原始的scale为5000）
		depth.data = depth_buffer.ptr<ushort>();
		depth.cols = depth_buffer.cols;
		depth.rows = depth_buffer.rows;
		depth.step = depth_buffer.cols*depth.elemSize(); // 1280 = 640*2

		if(imageMat.empty())
		{
			std::cout << "color_file empty!" << std::endl;
			return false; 
		}
			
		cv::cvtColor(imageMat, rgb_buffer, CV_BGR2RGB);

		rgb24.data = rgb_buffer.ptr<pcl::gpu::kinfuLS::PixelRGB>();
		rgb24.cols = rgb_buffer.cols;
		rgb24.rows = rgb_buffer.rows;
		rgb24.step = rgb_buffer.cols*sizeof(unsigned char);

		return true;
	}
	
	bool ImageGrab (string depth_file, string color_file, cv::Mat &depthMat, cv::Mat &imageMat, PtrStepSz<const unsigned short>& depth, PtrStepSz<const pcl::gpu::kinfuLS::PixelRGB>& rgb24)
	{
		depthMat = cv::imread(depth_file, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
		if(depthMat.empty())
		{
			std::cout << depth_file << " empty!" << std::endl;
			return false;
		}
		
		if (depthMat.elemSize() != sizeof(unsigned short))
		{
			cout << "Image was not opend in 16-bit format. Please use OpenCV 2.3.1 or higher" << endl;
			exit(1);
		}

		// Datasets are with factor 5000 (pixel to m) 
		// http://cvpr.in.tum.de/data/datasets/rgbd-dataset/file_formats#color_images_and_depth_maps
			
		depthMat.convertTo(impl_->depth_buffer, depthMat.type(), 0.2);
		depth.data = impl_->depth_buffer.ptr<ushort>();
		depth.cols = impl_->depth_buffer.cols;
		depth.rows = impl_->depth_buffer.rows;
		depth.step = impl_->depth_buffer.cols*depth.elemSize(); // 1280 = 640*2

		imageMat = cv::imread(color_file);
		if(imageMat.empty())
		{
			std::cout << color_file << " empty!" << std::endl;
			return false; 
		}
			
		cv::cvtColor(imageMat, impl_->rgb_buffer, CV_BGR2RGB);
		
		rgb24.data = impl_->rgb_buffer.ptr<pcl::gpu::kinfuLS::PixelRGB>();
		rgb24.cols = impl_->rgb_buffer.cols;
		rgb24.rows = impl_->rgb_buffer.rows;
		rgb24.step = impl_->rgb_buffer.cols*sizeof(unsigned char);

		return true;  
	}
      
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  void
  writeCloud (int format) const
  {      
    const SceneCloudView& view = scene_cloud_view_;

    if (!view.cloud_ptr_->points.empty ())
    {
      if(view.point_colors_ptr_->points.empty()) // no colors
      {
        if (view.valid_combined_)
          writeCloudFile (format, view.combined_ptr_);
        else
          writeCloudFile (format, view.cloud_ptr_);
      }
      else
      {        
        if (view.valid_combined_)
          writeCloudFile (format, merge<PointXYZRGBNormal>(*view.combined_ptr_, *view.point_colors_ptr_));
        else
          writeCloudFile (format, merge<PointXYZRGB>(*view.cloud_ptr_, *view.point_colors_ptr_));
      }
    }
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  void
  writeMesh(int format) const
  {
    if (scene_cloud_view_.mesh_ptr_) 
      writePolygonMeshFile(format, *scene_cloud_view_.mesh_ptr_);
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  void
  printHelp ()
  {
    cout << endl;
    cout << "KinFu app hotkeys" << endl;
    cout << "=================" << endl;
    cout << "    H    : print this help" << endl;
    cout << "   Esc   : exit" << endl;
    cout << "    T    : take cloud" << endl;
    cout << "    A    : take mesh" << endl;
    cout << "    M    : toggle cloud exctraction mode" << endl;
    cout << "    N    : toggle normals exctraction" << endl;
    cout << "    I    : toggle independent camera mode" << endl;
    cout << "    B    : toggle volume bounds" << endl;
    cout << "    *    : toggle scene view painting ( requires registration mode )" << endl;
    cout << "    C    : clear clouds" << endl;    
    cout << "   1,2,3 : save cloud to PCD(binary), PCD(ASCII), PLY(ASCII)" << endl;
    cout << "    7,8  : save mesh to PLY, VTK" << endl;
    cout << "   X, V  : TSDF volume utility" << endl;
    cout << "   L, l  : On the next shift, KinFu will extract the whole current cube, extract the world and stop" << endl;
    cout << "   S, s  : On the next shift, KinFu will extract the world and stop" << endl;
    cout << endl;
  }  

  bool exit_;
  bool scan_;
  bool scan_mesh_;
  bool scan_volume_;

  bool independent_camera_;
  int frame_counter_;
  pcl::kinfuLS::ScreenshotManager screenshot_manager_;
  int snapshot_rate_;

  bool registration_;
  bool integrate_colors_;
  bool pcd_source_;
  float focal_length_;

  KinfuTracker *kinfu_;
  ORB_SLAM2::System *orbslam2_;

  SceneCloudView scene_cloud_view_;
  ImageView image_view_;
  boost::shared_ptr<CurrentFrameCloudView> current_frame_cloud_view_;

  KinfuTracker::DepthMap depth_device_;

  pcl::PointCloud<pcl::PointXYZI>::Ptr tsdf_cloud_ptr_;

  Evaluation::Ptr evaluation_ptr_;

  boost::mutex data_ready_mutex_;
  boost::condition_variable data_ready_cond_;

  std::vector<pcl::gpu::kinfuLS::PixelRGB> source_image_data_;
  std::vector<unsigned short> source_depth_data_;
  PtrStepSz<const unsigned short> depth_;
  PtrStepSz<const pcl::gpu::kinfuLS::PixelRGB> rgb24_;  
  
  Eigen::Affine3f delta_lost_pose_;
  
  bool was_lost_;

  int time_ms_;
  
  boost::shared_ptr<Impl> impl_;

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  static void
  keyboard_callback (const visualization::KeyboardEvent &e, void *cookie)
  {
    KinFuLSApp* app = reinterpret_cast<KinFuLSApp*> (cookie);

    int key = e.getKeyCode ();

    if (e.keyUp ())    
      switch (key)
      {
      case 27: app->exit_ = true; break;
      case (int)'t': case (int)'T': app->scan_ = true; break;
      case (int)'a': case (int)'A': app->scan_mesh_ = true; break;
      case (int)'h': case (int)'H': app->printHelp (); break;
      case (int)'m': case (int)'M': app->scene_cloud_view_.toggleExtractionMode (); break;
      case (int)'n': case (int)'N': app->scene_cloud_view_.toggleNormals (); break;      
      case (int)'c': case (int)'C': app->scene_cloud_view_.clearClouds (true); break;
      case (int)'i': case (int)'I': app->toggleIndependentCamera (); break;
      case (int)'b': case (int)'B': app->scene_cloud_view_.toggleCube(app->kinfu_->volume().getSize()); break;
      case (int)'l': case (int)'L': app->kinfu_->performLastScan (); break;
      case (int)'s': case (int)'S': app->kinfu_->extractAndSaveWorld (); break;
      case (int)'r': case (int)'R': app->kinfu_->reset(); app->scene_cloud_view_.clearClouds(); break;
      case (int)'7': case (int)'8': app->writeMesh (key - (int)'0'); break;  
      case (int)'1': case (int)'2': case (int)'3': app->writeCloud (key - (int)'0'); break;      
      case '*': app->image_view_.toggleImagePaint (); break;
      
      case (int)'p': case (int)'P': app->kinfu_->setDisableICP(); break;

      case (int)'x': case (int)'X':
        app->scan_volume_ = !app->scan_volume_;
        cout << endl << "Volume scan: " << (app->scan_volume_ ? "enabled" : "disabled") << endl << endl;
        break;
      case (int)'v': case (int)'V':
        cout << "Saving TSDF volume to tsdf_volume.dat ... " << flush;
        // app->tsdf_volume_.save ("tsdf_volume.dat", true);
        app->kinfu_->volume ().save ("tsdf_volume.dat", true);
        // cout << "done [" << app->tsdf_volume_.size () << " voxels]" << endl;
        cout << "done [" << app->kinfu_->volume ().size () << " voxels]" << endl;
        cout << "Saving TSDF volume cloud to tsdf_cloud.pcd ... " << flush;
        pcl::io::savePCDFile<pcl::PointXYZI> ("tsdf_cloud.pcd", *app->tsdf_cloud_ptr_, true);
        cout << "done [" << app->tsdf_cloud_ptr_->size () << " points]" << endl;
        break;
      default:
        break;
      }    
  }

};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename CloudPtr> void
writeCloudFile (int format, const CloudPtr& cloud_prt)
{
  if (format == KinFuLSApp::PCD_BIN)
  {
    cout << "Saving point cloud to 'cloud_bin.pcd' (binary)... " << flush;
    pcl::io::savePCDFile ("cloud_bin.pcd", *cloud_prt, true);
  }
  else if (format == KinFuLSApp::PCD_ASCII)
  {
    cout << "Saving point cloud to 'cloud.pcd' (ASCII)... " << flush;
    pcl::io::savePCDFile ("cloud.pcd", *cloud_prt, false);
  }
  else   /* if (format == KinFuLSApp::PLY) */
  {
    cout << "Saving point cloud to 'cloud.ply' (ASCII)... " << flush;
    pcl::io::savePLYFileASCII ("cloud.ply", *cloud_prt);

  }
  cout << "Done" << endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
writePolygonMeshFile (int format, const pcl::PolygonMesh& mesh)
{
  if (format == KinFuLSApp::MESH_PLY)
  {
    cout << "Saving mesh to to 'mesh.ply'... " << flush;
    pcl::io::savePLYFile("mesh.ply", mesh);		
  }
  else /* if (format == KinFuLSApp::MESH_VTK) */
  {
    cout << "Saving mesh to to 'mesh.vtk'... " << flush;
    pcl::io::saveVTKFile("mesh.vtk", mesh);    
  }  
  cout << "Done" << endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int
print_cli_help ()
{
	cout << "\nKinFu parameters:" << endl;
	cout << "    --help, -h                          : print this message" << endl;  
	cout << "    --integrate-colors, -ic             : enable color integration mode (allows to get cloud with colors)" << endl;
	cout << endl << "";
	cout << " For RGBD benchmark (default):" << endl; 
	cout << "    -eval <eval_folder>" << endl << endl;

	return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void loadImages(const string databasePath, const string associatedFile, vector<double> &vTimestamps, vector<string> &vstrRGB, vector<string> &vstrDepth)
{
	ifstream fAssociation(associatedFile);
	if(!fAssociation.is_open())
	{
		cout << associatedFile << " dose not exist!" << endl;
		exit(0);
	}
	
	while(!fAssociation.eof())
	{
		string s;
		getline(fAssociation, s);
		if(!s.empty())
		{
			stringstream ss;
			ss << s;
			
			double t;
			string sRGB, sD;
			
			ss >> t;
			vTimestamps.push_back(t);
			ss >> sRGB;
			vstrRGB.push_back(databasePath + "/" + sRGB);
			ss >> t;
			ss >> sD;
			vstrDepth.push_back(databasePath + "/" + sD);
		}
	}
	
	cout << "Total number of images = " << vstrRGB.size() << endl;
}

int main(int argc, char *argv[])
{
	if(argc != 5)
	{
		cout << "Usage: ./orbslam2TSDF path_to_vocabulary path_to_settings database_path image_associated_file" << endl;
		return -1;
	}
	
	string vocabularyPath(argv[1]);
	string settingPath(argv[2]);
	string databasePath(argv[3]);
	string associatedFile(argv[4]);
	
	vector<double> vTimestamps;
	vector<string> vstrRGB;
	vector<string> vstrDepth;
	
	cout << "vocabulary path = " << vocabularyPath << endl;
	cout << "setting path = " << settingPath << endl;
	cout << "databasePath = " << databasePath << endl;
	cout << "associatedFile = " << associatedFile << endl;
	loadImages(databasePath, associatedFile, vTimestamps, vstrRGB, vstrDepth);
	
	
	if (pc::find_switch (argc, argv, "--help") || pc::find_switch (argc, argv, "-h"))
		return print_cli_help ();

	float volume_size = pcl::device::kinfuLS::VOLUME_SIZE;
	float shift_distance = pcl::device::kinfuLS::DISTANCE_THRESHOLD;
	int snapshot_rate = pcl::device::kinfuLS::SNAPSHOT_RATE;
	
	KinFuLSApp app (volume_size, shift_distance, snapshot_rate, vocabularyPath, settingPath);
	
	if (pc::find_switch (argc, argv, "--integrate_colors") || pc::find_switch (argc, argv, "-ic"))
	{
		app.toggleColorIntegration_database();
	}
	
	app.initCurrentFrameView();
	
	try
	{
		app.startMainLoop_use_database(vTimestamps, vstrRGB, vstrDepth);
	}
	catch (const pcl::PCLException& /*e*/) { cout << "PCLException" << endl; }
	catch (const std::bad_alloc& /*e*/) { cout << "Bad alloc" << endl; }
	catch (const std::exception& /*e*/) { cout << "Exception" << endl; }
	
	return 0;
}