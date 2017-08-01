#include <vector>

//uima
#include <uima/api.hpp>
#include <uima/fsfilterbuilder.hpp>

//
#include <opencv2/opencv.hpp>

//pcl
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/common/centroid.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/impl/kdtree.hpp>

//project
#include <rs/scene_cas.h>
#include <rs/DrawingAnnotator.h>
#include <rs/utils/time.h>
#include <rs/utils/output.h>
#include <rs/utils/common.h>

//#define DEBUG_OUTPUT 1;

using namespace uima;

/**
 * @brief The PointCloudClusterExtractor class
 * given a plane equation,
 */
class TangoCloudClusterExtractor : public DrawingAnnotator
{
private:
    typedef pcl::PointXYZRGBA PointT;

    struct Cluster
    {
        size_t indicesIndex;
        cv::Rect roi, roiHires;
        cv::Mat mask, maskHires;
    };

    Type cloud_type;
    cv::Mat color;
    int cluster_min_size, cluster_max_size;
    float polygon_min_height, polygon_max_height;
    float cluster_tolerance;
    float plane_distance_threshold;
    pcl::PointCloud<PointT>::Ptr cloud_ptr;
    std::vector<pcl::PointIndices> cluster_indices;
    std::vector<Cluster> clusters;
    double pointSize;
    Eigen::Matrix3d K;
    Eigen::Vector3d D;

public:

    TangoCloudClusterExtractor(): DrawingAnnotator(__func__), cluster_min_size(50), cluster_max_size(2500),
        polygon_min_height(0.03), polygon_max_height(0.5), cluster_tolerance(0.02), pointSize(1)
    {
        cloud_ptr = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
    }

    TyErrorId initialize(AnnotatorContext &ctx)
    {
        outInfo("initialize");

        return UIMA_ERR_NONE;
    }

    TyErrorId destroy()
    {
        outInfo("destroy");
        return UIMA_ERR_NONE;
    }

private:

    TyErrorId processWithLock(CAS &tcas, ResultSpecification const &res_spec)
    {
        MEASURE_TIME;
        outInfo("process begins");
        rs::StopWatch clock;
        double t = clock.getTime();

        rs::SceneCas cas(tcas);
        rs::Scene scene = cas.getScene();

        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
        pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices());
        pcl::PointIndices::Ptr prism_inliers(new pcl::PointIndices());
        cluster_indices.clear();

        cas.get(VIEW_CLOUD, *cloud_ptr);
        cas.get(VIEW_NORMALS, *cloud_normals);
        cas.get(VIEW_COLOR_IMAGE, color);

        sensor_msgs::CameraInfo cameraInfo;
        cas.get(VIEW_COLOR_CAMERA_INFO, cameraInfo);
        readCameraInfo(cameraInfo);

        std::vector<rs::Plane> planes;
        scene.annotations.filter(planes);
        if(planes.empty())
        {
            outInfo("NO PLANE COEFFICIENTS SET!! RUN A PLANE ESIMTATION BEFORE!!!");
            outInfo(clock.getTime() << " ms.");
            return UIMA_ERR_ANNOTATOR_MISSING_INFO;
        }

        plane_coefficients->values = planes[0].model();
        plane_inliers->indices = planes[0].inliers();

        if(plane_coefficients->values.empty())
        {
            outInfo("PLane COEFFICIENTS EMPTY");
            outInfo(clock.getTime() << " ms.");
            return UIMA_ERR_NONE;
        }
        outDebug("getting input data took : " << clock.getTime() - t << " ms.");
        t = clock.getTime();

        cloudPreProcessing(cloud_ptr, plane_coefficients, plane_inliers, prism_inliers);
        outDebug("cloud preprocessing took : " << clock.getTime() - t << " ms.");
        t = clock.getTime();
        pointCloudClustering(cloud_ptr, prism_inliers, cluster_indices);

        clusters.resize(cluster_indices.size());

#pragma omp parallel for schedule(dynamic)
        for(size_t i = 0; i < cluster_indices.size(); ++i)
        {
            Cluster &cluster = clusters[i];
            cluster.indicesIndex = i;

            createImageRoi(cluster);
        }
        outDebug("conversion to image ROI took: " << clock.getTime() - t << " ms.");
        t = clock.getTime();

        for(size_t i = 0; i < cluster_indices.size(); ++i)
        {
            Cluster &cluster = clusters[i];
            const pcl::PointIndices &indices = cluster_indices[i];

            rs::Cluster uimaCluster = rs::create<rs::Cluster>(tcas);
            rs::ReferenceClusterPoints rcp = rs::create<rs::ReferenceClusterPoints>(tcas);
            rs::PointIndices uimaIndices = rs::conversion::to(tcas, indices);

            //outDebug("cluster size: " << indices.indices.size());
            rcp.indices.set(uimaIndices);

            rs::ImageROI imageRoi = rs::create<rs::ImageROI>(tcas);
            imageRoi.mask(rs::conversion::to(tcas, cluster.mask));
            imageRoi.mask_hires(rs::conversion::to(tcas, cluster.maskHires));
            imageRoi.roi(rs::conversion::to(tcas, cluster.roi));
            imageRoi.roi_hires(rs::conversion::to(tcas, cluster.roiHires));

            uimaCluster.points.set(rcp);
            uimaCluster.rois.set(imageRoi);
            uimaCluster.source.set("EuclideanClustering");
            scene.identifiables.append(uimaCluster);
        }
        outDebug("adding clusters took: " << clock.getTime() - t << " ms.");

        return UIMA_ERR_NONE;
    }

    void drawImageWithLock(cv::Mat &disp)
    {
        disp = color.clone();
        for(size_t i = 0; i < clusters.size(); ++i)
        {
            cv::rectangle(disp, clusters[i].roi, rs::common::cvScalarColors[i % rs::common::numberOfColors]);
        }
    }

    void fillVisualizerWithLock(pcl::visualization::PCLVisualizer &visualizer, const bool firstRun)
    {
        const std::string &cloudname = this->name;
        for(size_t i = 0; i < cluster_indices.size(); ++i)
        {
            const pcl::PointIndices &indices = cluster_indices[i];
            for(size_t j = 0; j < indices.indices.size(); ++j)
            {
                size_t index = indices.indices[j];
                cloud_ptr->points[index].rgba = rs::common::colors[i % rs::common::numberOfColors];
            }
        }

        if(firstRun)
        {
            visualizer.addPointCloud(cloud_ptr, cloudname);
            visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pointSize, cloudname);
        }
        else
        {
            visualizer.updatePointCloud(cloud_ptr, cloudname);
            visualizer.getPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pointSize, cloudname);
        }
    }

    void cloudPreProcessing(const pcl::PointCloud<PointT>::Ptr &cloud,
                            const pcl::ModelCoefficients::Ptr &plane_coeffs,
                            const pcl::PointIndices::Ptr &plane_inliers,
                            pcl::PointIndices::Ptr &prism_inliers)
    {
        pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>);
        pcl::ExtractIndices<PointT> ei;
        ei.setInputCloud(cloud);
        ei.setIndices(plane_inliers);
        ei.filter(*cloud_plane);

        // Get convex hull
        pcl::PointCloud<PointT>::Ptr cloud_hull(new pcl::PointCloud<PointT>);
        pcl::ConvexHull<PointT> chull;
        chull.setInputCloud(cloud_plane);
        chull.reconstruct(*cloud_hull);

        outDebug(" Convex hull has: " << cloud_hull->points.size() << " data points.");

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cloud_hull, centroid);
        double hull_shrink_factor = 0;

        for(size_t i = 0; i < cloud_hull->points.size(); ++i)
        {
            Eigen::Vector4f scaled_vector = (cloud_hull->points[i].getVector4fMap() - centroid) * hull_shrink_factor;
            cloud_hull->points[i].getVector4fMap() -= scaled_vector;
        }

        // Get points in polygonal prism
        pcl::ExtractPolygonalPrismData<PointT> epm;
        epm.setInputCloud(cloud);
        epm.setInputPlanarHull(cloud_hull);

        /*TODO::Check why this is so sensitive
    why does this need to be so freaking high? (if set lower it finds
    points that are actually part of the plane)*/
        epm.setHeightLimits(polygon_min_height, polygon_max_height);
        epm.segment(*prism_inliers);
        outDebug("points in the prism: " << prism_inliers->indices.size());
    }

    bool pointCloudClustering(const pcl::PointCloud<PointT>::Ptr &cloud,
                              const pcl::PointIndices::Ptr &indices,
                              std::vector<pcl::PointIndices> &cluster_indices)
    {

        if(indices->indices.size() > 0)
        {
            pcl::EuclideanClusterExtraction<PointT> ec;
            pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
            tree->setInputCloud(cloud, boost::make_shared<std::vector<int>>(indices->indices));
            ec.setClusterTolerance(cluster_tolerance);
            ec.setSearchMethod(tree);
            ec.setInputCloud(cloud);
            ec.setIndices(indices);
            ec.setMinClusterSize(cluster_min_size);
            ec.extract(cluster_indices);
            return true;
        }
        else
        {
            return false;
        }
    }

    /**
   * given orignal_image and reference cluster points, compute an image containing only the cluster
   */
    void createImageRoi(Cluster &cluster) const
    {
        const pcl::PointIndices &indices = cluster_indices[cluster.indicesIndex];

        size_t width = color.size().width;
        size_t height = color.size().height;

        int min_x = width;
        int max_x = -1;
        int min_y = height;
        int max_y = -1;

        cv::Mat mask_full = cv::Mat::zeros(height, width, CV_8U);

        for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
        {
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGBA>);
            for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
                cloud_cluster->points.push_back (cloud_ptr->points[*pit]); //*
            cloud_cluster->width = cloud_cluster->points.size();
            cloud_cluster->height = 1;
            cloud_cluster->is_dense = true;

            if(cloud_cluster->points.size() == indices.indices.size())
            {
                for(size_t i = 0; i < indices.indices.size(); ++i)//for (size_t j = 0; j < indices.indices.size(); j++)
                {
                    Eigen::Vector2d imageCoords;
                    imageCoords[0] = cloud_cluster->points[i].x/cloud_cluster->points[i].z;
                    imageCoords[1] = cloud_cluster->points[i].y/cloud_cluster->points[i].z;

//                    cv::Mat img = color.clone();
//                    mask_full = cv::Mat::zeros(img.size().height, img.size().width, CV_8U);

                    float r2 = imageCoords.adjoint()*imageCoords;
                    float r4 = r2*r2;
                    float r6 = r2*r4;

                    imageCoords = imageCoords*(1.0 + D[0]*r2 + D[1]*r4 + D[2]*r6);

                    Eigen::Vector3d imageCoords_3;
                    imageCoords_3[0]=imageCoords[0];
                    imageCoords_3[1]=imageCoords[1];
                    imageCoords_3[2]=1;

                    Eigen::Vector3d pixelCoords;
                    pixelCoords = K*imageCoords_3;

                    pixelCoords[0] = static_cast<unsigned int>(pixelCoords[0]);
                    pixelCoords[1] = static_cast<unsigned int>(pixelCoords[1]);

                    const int x = pixelCoords[0];
                    const int y = pixelCoords[1];

                    min_x = std::min(min_x, x);
                    min_y = std::min(min_y, y);
                    max_x = std::max(max_x, x);
                    max_y = std::max(max_y, y);

                    mask_full.at<uint8_t>(y, x) = 255;
                }
            }
        }
        cluster.roi = cv::Rect(min_x-10, min_y-10, max_x - min_x + 10, max_y - min_y + 10);
        cluster.roiHires = cv::Rect(cluster.roi.x << 1, cluster.roi.y << 1, cluster.roi.width << 1, cluster.roi.height << 1);
        mask_full(cluster.roi).copyTo(cluster.mask);
        cv::resize(cluster.mask, cluster.maskHires, cv::Size(0, 0), 2.0, 2.0, cv::INTER_NEAREST);
    }
    void readCameraInfo(const sensor_msgs::CameraInfo &camInfo)
    {
        K << camInfo.K[0], camInfo.K[1], camInfo.K[2],
                camInfo.K[3], camInfo.K[4], camInfo.K[5],
                camInfo.K[6], camInfo.K[7], camInfo.K[8];
        D[0]=camInfo.D[0];
        D[1]=camInfo.D[1];
        D[2]=camInfo.D[2];
    }
};
// This macro exports an entry point that is used to create the annotator.
MAKE_AE(TangoCloudClusterExtractor)
