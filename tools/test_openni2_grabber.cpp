/*************************************************************************
    > File Name: test_openni2_grabber.cpp
    > Author: QinZiwen
    > Mail: qinziwen2013@163.com 
    > Created Time: 2017年04月07日 星期五 09时40分00秒
 ************************************************************************/

#include<iostream>
using namespace std;

#include <pcl/io/openni_grabber.h>
#include <pcl/io/openni2_grabber.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/oni_grabber.h>

class SimpleOpenNIViewer
{
public:
    SimpleOpenNIViewer() : viewer("PCL OpenNI Viewer") 
	{
		fx = 517.306408;
		fy = 516.469215;
		cx = 318.643040;
		cy = 255.313989;
	}
	~SimpleOpenNIViewer()
	{}
    
    // 定义回调函数cloud_cb_,获取到数据时对数据进行处理
    void cloud_cb_(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud)
    {
        if (!viewer.wasStopped()) // Check if the gui was quit. true if the user signaled the gui to stop
            viewer.showCloud(cloud);
    }
    
    void cloud_cb_rgb(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &cloud)
    {
        if (!viewer.wasStopped()) // Check if the gui was quit. true if the user signaled the gui to stop
            viewer.showCloud(cloud);
    }

	//void (const boost::shared_ptr<openni_wrapper::DepthImage>&)
	//void source_cb1(const boost::shared_ptr<openni2_wrapper::DepthImage>& depth_wrapper)
	//void (sig_cb_openni_depth_image) (const boost::shared_ptr<DepthImage>&)
	void source_cb1(const boost::shared_ptr<pcl::io::DepthImage>& depth_wrapper)  
	{
		PCL_WARN("RUN source_cb1 ...\n");
		
		cout << "depth_wrapper->getWidth() = " << depth_wrapper->getWidth() << endl;
		cout << "depth_wrapper->getHeight() = " << depth_wrapper->getHeight() << endl;
	}

    void run()
    {
        // create a new grabber for OpenNI devices
        pcl::Grabber* interface = new pcl::io::OpenNI2Grabber();
		//pcl::io::OpenNI2Grabber *interface = new pcl::io::OpenNI2Grabber();

        // make callback function from member function
        //boost::function<void(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr&)> f = boost::bind(&SimpleOpenNIViewer::cloud_cb_, this, _1);
		//boost::function<void(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)> f = boost::bind(&SimpleOpenNIViewer::cloud_cb_rgb, this, _1);
		//boost::function<void(const boost::shared_ptr<openni2_wrapper::DepthImage>&)> f = boost::bind(&SimpleOpenNIViewer::source_cb1, this, _1);
		boost::function<void(const boost::shared_ptr<pcl::io::DepthImage>&)> f = boost::bind(&SimpleOpenNIViewer::source_cb1, this, _1);

        // connect callback function for desired signal
        boost::signals2::connection c = interface->registerCallback(f);

		//interface->setRGBCameraIntrinsics(fx,fy,cx,cy);
		//interface->isRunning();
        // start receiving point clouds
        interface->start();

        while (!viewer.wasStopped())
        {
            boost::this_thread::sleep(boost::posix_time::seconds(1));
        }

        // Stop the data acquisition
        interface->stop();
    }

    pcl::visualization::CloudViewer viewer;
	double fx;
	double fy;
	double cx;
	double cy;
};


int main()
{
	//boost::shared_ptr<pcl::Grabber> capture;
	//capture.reset( new pcl::io::OpenNI2Grabber() );	
	
	pcl::Grabber *grab = new pcl::io::OpenNI2Grabber();
	grab->start();
	grab->stop();
	
    //SimpleOpenNIViewer v;
    //v.run();
    
    return 0;
}