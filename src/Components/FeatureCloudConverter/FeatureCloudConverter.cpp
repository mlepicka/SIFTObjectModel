/*!
 * \file
 * \brief
 * \author Micha Laszkowski
 */

#include <memory>
#include <string>

#include "FeatureCloudConverter.hpp"
#include "Common/Logger.hpp"

#include <boost/bind.hpp>

namespace Processors {
namespace FeatureCloudConverter {

FeatureCloudConverter::FeatureCloudConverter(const std::string & name) :
		Base::Component(name)  {

}

FeatureCloudConverter::~FeatureCloudConverter() {
}

void FeatureCloudConverter::prepareInterface() {
	// Register data streams, events and event handlers HERE!
	registerStream("in_depth", &in_depth);
	registerStream("in_mask", &in_mask);
	registerStream("in_features", &in_features);
	registerStream("in_descriptors", &in_descriptors);
	registerStream("in_camera_info", &in_camera_info);
    registerStream("in_depth_xyz", &in_depth_xyz);
	registerStream("out_cloud_xyzsift", &out_cloud_xyzsift);
	registerStream("out_cloud_xyzkaze", &out_cloud_xyzkaze);

	// Register handlers
	registerHandler("process", boost::bind(&FeatureCloudConverter::process, this));
	addDependency("process", &in_depth);
	addDependency("process", &in_features);
	addDependency("process", &in_descriptors);
	addDependency("process", &in_camera_info);

	registerHandler("process_mask", boost::bind(&FeatureCloudConverter::process_mask, this));
	addDependency("process_mask", &in_depth);
	addDependency("process_mask", &in_mask);
	addDependency("process_mask", &in_features);
	addDependency("process_mask", &in_descriptors);
	addDependency("process_mask", &in_camera_info);

	registerHandler("process_depth_xyz", boost::bind(&FeatureCloudConverter::process_depth_xyz, this));
    addDependency("process_depth_xyz", &in_features);
    addDependency("process_depth_xyz", &in_descriptors);
    addDependency("proces_depth_xyz", &in_depth_xyz);

	registerHandler("process_depth_xyz_mask", boost::bind(&FeatureCloudConverter::process_depth_xyz_mask, this));
    addDependency("process_depth_xyz_mask", &in_mask);
    addDependency("process_depth_xyz_mask", &in_features);
    addDependency("process_depth_xyz_mask", &in_descriptors);
    addDependency("process_depth_xyz_mask", &in_depth_xyz);

}

bool FeatureCloudConverter::onInit() {

	return true;
}

bool FeatureCloudConverter::onFinish() {
	return true;
}

bool FeatureCloudConverter::onStop() {
	return true;
}

bool FeatureCloudConverter::onStart() {
	return true;
}

template<typename T> void FeatureCloudConverter::create_cloud(const Types::Features& features,
		cv::Mat depth, double cx_d, double fx_d, double cy_d, double fy_d,
		const cv::Mat& descriptors,
		const typename  pcl::PointCloud<T>::Ptr& cloud) {
	for (int i = 0; i < features.features.size(); i++) {
		T point;
		int u = round(features.features[i].pt.x);
		int v = round(features.features[i].pt.y);
		float d = depth.at<float>(v, u);
		if (d == 0) {
			continue;
		}
		point.x = (u - cx_d) * d * fx_d;
		point.y = (v - cy_d) * d * fy_d;
		point.z = d * 0.001;
		for (int j = 0; j < descriptors.cols; j++) {
			point.descriptor[j] = descriptors.row(i).at<float>(j);
		}
		point.multiplicity = 1;
		cloud->push_back(point);
	}
}

template<typename T> void FeatureCloudConverter::create_cloud_mask(const Types::Features& features,
		cv::Mat depth, double cx_d, double fx_d, double cy_d, double fy_d,
		const cv::Mat& descriptors,
		const typename  pcl::PointCloud<T>::Ptr& cloud, cv::Mat mask) {
		for(int i=0; i < features.features.size(); i++){
			
			T point;
			int u = round(features.features[i].pt.x);
			int v = round(features.features[i].pt.y);

			float d = depth.at<float>(v, u);
			if (d == 0 || mask.at<float>(v, u)==0) {
					continue;
			}
			
			point.x = (u - cx_d) * d * fx_d;
			point.y = (v - cy_d) * d * fy_d;
			point.z = d * 0.001;

			for(int j=0; j<descriptors.cols;j++){
				point.descriptor[j] = descriptors.row(i).at<float>(j);	
			}
			
			point.multiplicity = 1;
			
			cloud->push_back(point);
	}
}

template<typename T> void FeatureCloudConverter::create_cloud_depth_xyz_mask(const Types::Features& features, cv::Mat depth_xyz,
			const cv::Mat& descriptors,
			const typename pcl::PointCloud<T>::Ptr& cloud, cv::Mat mask){
				    const double max_z = 1.0e4;

    for(int i=0; i < features.features.size(); i++){
        T point;
        int u = round(features.features[i].pt.x);
        int v = round(features.features[i].pt.y);
        if (mask.at<float>(v, u)==0) {
                continue;
        }
        
        cv::Vec3f p = depth_xyz.at<cv::Vec3f>(v, u);
        if(fabs(p[2] - max_z) < FLT_EPSILON || fabs(p[2]) > max_z) continue;

       
        point.x = p[0];
        point.y = p[1];
        point.z = p[2];

        for(int j=0; j<descriptors.cols;j++){
            point.descriptor[j] = descriptors.row(i).at<float>(j);
        }

        point.multiplicity = 1;
        cloud->push_back(point);
    }
}


template<typename T> void FeatureCloudConverter::create_cloud_depth_xyz(const Types::Features& features, cv::Mat depth_xyz,
			const cv::Mat& descriptors,
			const typename pcl::PointCloud<T>::Ptr& cloud){
				const double max_z = 1.0e4;

    for(int i=0; i < features.features.size(); i++){

        T point;
        int u = round(features.features[i].pt.x);
        int v = round(features.features[i].pt.y);

        cv::Vec3f p = depth_xyz.at<cv::Vec3f>(v, u);
        if(fabs(p[2] - max_z) < FLT_EPSILON || fabs(p[2]) > max_z) continue;

        point.x = p[0];
        point.y = p[1];
        point.z = p[2];


        for(int j=0; j<descriptors.cols;j++){
            point.descriptor[j] = descriptors.row(i).at<float>(j);
        }

        point.multiplicity = 1;

        cloud->push_back(point);
    }
		
}

void FeatureCloudConverter::process() {
	CLOG(LTRACE) << "FeatureCloudConverter::process";
	cv::Mat depth = in_depth.read();
	depth.convertTo(depth, CV_32F);
	cv::Mat descriptors = in_descriptors.read();
	Types::Features features = in_features.read();
	Types::CameraInfo camera_info = in_camera_info.read();
			
	double fx_d = 0.001 / camera_info.fx();
	double fy_d = 0.001 / camera_info.fy();
	double cx_d = camera_info.cx();
	double cy_d = camera_info.cy();
	
	CLOG(LTRACE) << "Feature type: "<< features.type;
	if(features.type=="SIFT"){
		CLOG(LTRACE) << "Create SIFT cloud!";
		pcl::PointCloud<PointXYZSIFT>::Ptr cloud (new pcl::PointCloud<PointXYZSIFT>());
		create_cloud<PointXYZSIFT>(features, depth, cx_d, fx_d, cy_d, fy_d, descriptors, cloud);
		out_cloud_xyzsift.write(cloud);
	}
	else if(features.type=="KAZE"){
		CLOG(LTRACE) << "Create KAZE cloud!";
		pcl::PointCloud<PointXYZKAZE>::Ptr cloud (new pcl::PointCloud<PointXYZKAZE>());
		create_cloud<PointXYZKAZE>(features, depth, cx_d, fx_d, cy_d, fy_d, descriptors, cloud);
		out_cloud_xyzkaze.write(cloud);
	}
}


void FeatureCloudConverter::process_mask() {
	CLOG(LTRACE) << "FeatureCloudConverter::process_mask";
	cv::Mat depth = in_depth.read();
	depth.convertTo(depth, CV_32F);
	cv::Mat mask = in_mask.read();
	mask.convertTo(mask, CV_32F);
	cv::Mat descriptors = in_descriptors.read();
	Types::Features features = in_features.read();
	Types::CameraInfo camera_info = in_camera_info.read();

	double fx_d = 0.001 / camera_info.fx();
	double fy_d = 0.001 / camera_info.fy();
	double cx_d = camera_info.cx();
	double cy_d = camera_info.cy();

	CLOG(LTRACE) << "Feature type: "<< features.type;
	if(features.type=="SIFT"){
		CLOG(LTRACE) << "Create SIFT cloud!";
		pcl::PointCloud<PointXYZSIFT>::Ptr cloud (new pcl::PointCloud<PointXYZSIFT>());
		create_cloud_mask<PointXYZSIFT>(features, depth, cx_d, fx_d, cy_d, fy_d, descriptors, cloud, mask);
		out_cloud_xyzsift.write(cloud);
	}
	else if(features.type=="KAZE"){
		CLOG(LTRACE) << "Create KAZE cloud!";
		pcl::PointCloud<PointXYZKAZE>::Ptr cloud (new pcl::PointCloud<PointXYZKAZE>());
		create_cloud_mask<PointXYZKAZE>(features, depth, cx_d, fx_d, cy_d, fy_d, descriptors, cloud, mask);
		out_cloud_xyzkaze.write(cloud);
	}
}

void FeatureCloudConverter::process_depth_xyz() {
    CLOG(LTRACE) << "FeatureCloudConverter::process_depth_xyz";
    cv::Mat depth_xyz = in_depth_xyz.read();
    cv::Mat descriptors = in_descriptors.read();
    Types::Features features = in_features.read();

	CLOG(LTRACE) << "Feature type: "<< features.type;
    if(features.type=="SIFT"){
		CLOG(LTRACE) << "Create SIFT cloud!";
		pcl::PointCloud<PointXYZSIFT>::Ptr cloud (new pcl::PointCloud<PointXYZSIFT>());
		create_cloud_depth_xyz<PointXYZSIFT>(features, depth_xyz, descriptors, cloud);
		out_cloud_xyzsift.write(cloud);
	}
	else if(features.type=="KAZE"){
		CLOG(LTRACE) << "Create KAZE cloud!";
		pcl::PointCloud<PointXYZKAZE>::Ptr cloud (new pcl::PointCloud<PointXYZKAZE>());
		create_cloud_depth_xyz<PointXYZKAZE>(features, depth_xyz, descriptors, cloud);
		out_cloud_xyzkaze.write(cloud);
	}
}

void FeatureCloudConverter::process_depth_xyz_mask() {
    CLOG(LTRACE) << "FeatureCloudConverter::process_depth_xyz_mask";
    cv::Mat depth_xyz = in_depth_xyz.read();
    cv::Mat descriptors = in_descriptors.read();
    Types::Features features = in_features.read();
    cv::Mat mask = in_mask.read();
    mask.convertTo(mask, CV_32F);
    
    CLOG(LTRACE) << "Feature type: "<< features.type.c_str();
  	if(features.type=="SIFT"){
		CLOG(LTRACE) << "Create SIFT cloud!";
		pcl::PointCloud<PointXYZSIFT>::Ptr cloud (new pcl::PointCloud<PointXYZSIFT>());
		create_cloud_depth_xyz_mask<PointXYZSIFT>(features, depth_xyz, descriptors, cloud, mask);
		out_cloud_xyzsift.write(cloud);
	}
	else if(features.type=="KAZE"){
		CLOG(LTRACE) << "Create KAZE cloud!";
		pcl::PointCloud<PointXYZKAZE>::Ptr cloud (new pcl::PointCloud<PointXYZKAZE>());
		create_cloud_depth_xyz_mask<PointXYZKAZE>(features, depth_xyz, descriptors, cloud, mask);
		out_cloud_xyzkaze.write(cloud);
	}
}

} //: namespace FeatureCloudConverter
} //: namespace Processors
