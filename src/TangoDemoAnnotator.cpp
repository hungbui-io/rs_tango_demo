#include <uima/api.hpp>

#include <pcl/point_types.h>
#include <rs/types/all_types.h>
#include <pcl_conversions/pcl_conversions.h>
//RS
#include <rs/scene_cas.h>
#include <rs/utils/time.h>

//ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
//#include <sensor_msgs/image_encodings.h>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>


using namespace uima;


class TangoDemoAnnotator : public Annotator
{
private:
    ros::NodeHandle nh_;

    image_transport::Publisher image_pub;
    //ros::Publisher image_pub;
    ros::Publisher cloud_pub;
    ros::Publisher camInfo_pub;
public:

//    TangoDemoAnnotator(): nh_("~")
//    {
//        image_pub = nh_.advertise<sensor_msgs::Image>("/rs/tango/image", 1);
//        cloud_pub = nh_.advertise<sensor_msgs::PointCloud2>("/rs//tango/cloud", 1);
//        camInfo_pub = nh_.advertise<sensor_msgs::CameraInfo>("/rs/tango/cameraInfo", 1);
//    }
  TyErrorId initialize(AnnotatorContext &ctx)
  {
      image_transport::ImageTransport it(nh_);
      image_pub = it.advertise("/rs/tango/image", 1);
      cloud_pub = nh_.advertise<sensor_msgs::PointCloud2>("/rs//tango/cloud", 1);
      camInfo_pub = nh_.advertise<sensor_msgs::CameraInfo>("/rs/tango/cameraInfo", 1);
      outInfo("initialize");
    return UIMA_ERR_NONE;
  }

  TyErrorId destroy()
  {
    outInfo("destroy");
    return UIMA_ERR_NONE;
  }

  TyErrorId process(CAS &tcas, ResultSpecification const &res_spec)
  {
      outInfo("process start");
      rs::StopWatch clock;
      rs::SceneCas cas(tcas);
      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGBA>);
      cv::Mat image;
      sensor_msgs::CameraInfo cameraInfo;

      cas.get(VIEW_CLOUD,*cloud_ptr);
      cas.get(VIEW_CLUSTER_IMAGE, image);
      cas.get(VIEW_COLOR_CAMERA_INFO, cameraInfo);

      outInfo("Cloud size: " << cloud_ptr->points.size());
      outInfo("took: " << clock.getTime() << " ms.");

      //outInfo("Image size: " << image.size());

      //sensor_msgs::CameraInfo camInfo_msg;


      sensor_msgs::PointCloud2 cloud_msg;
      sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
      pcl::toROSMsg(*cloud_ptr, cloud_msg);
      image_pub.publish(*image_msg);
      cloud_pub.publish(cloud_msg);
      camInfo_pub.publish(cameraInfo);
      outInfo("cloud width: " << cloud_msg.width);
      outInfo("image width: " << image_msg->width);
      return UIMA_ERR_NONE;
  }
};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(TangoDemoAnnotator)

