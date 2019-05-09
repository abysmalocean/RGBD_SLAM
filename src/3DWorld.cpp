#include <iostream>
#include <../include/utility.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <dirent.h>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>

#include <omp.h>


using std::cout; 
using std::endl;





int main( int argc, char** argv )
{
    //Read data
    ParameterReader pd;
    int display = atoi( pd.getData( "display" ).c_str() );
    int imgDisplay = atoi( pd.getData( "imageDisplay" ).c_str() );
    int width  =   atoi( pd.getData( "width" ).c_str() );
    int height    =   atoi( pd.getData( "height"   ).c_str() );

    std::string folder = pd.getData("DataFolder"); 
    cout << "Working on the folder --> [" << folder << "]<---" << endl; 
    

    int nFrames = 0;

	std::string imgsFolder = folder;
	int count = 0; 
	std::vector<std::string> paths; 
	if (auto dir = opendir(imgsFolder.c_str()))
	{
		while(auto f= readdir(dir))
		{
			if (!f->d_name || f->d_name[0] == '.')
                continue; // Skip everything that starts with a dot
			std::string filePath = imgsFolder + "/" + f->d_name; 
			//printf("File: %s | num [%zu]\n", f->d_name, count++);
			paths.push_back(filePath); 
			++count;
		}
	}
    

    std::vector<FRAME> source(count); 
    count = 0;
    std::sort(paths.begin(), paths.end()); 
    cout << "Loading the Source Image to memory" << endl; 
    #pragma omp parallel for
    for (int i = 0; i < paths.size(); ++i)
    {
        FRAME frame = readImage(paths[i], &pd);
        //frame.frameID = count++; 
        source[i] = frame; 

    }
    printf("Source finish \n");

    int nfeatures = atoi( pd.getData( "nfeatures" ).c_str() );
    int nOctaveLayers =  atoi( pd.getData( "nOctaveLayers" ).c_str() );
    double contrastThreshold = atoi(pd.getData( "contrastThreshold" ).c_str())/100.0; 
    double edgeThreshold  = atoi(pd.getData( "edgeThreshold" ).c_str()) * 1.0; 
    double sigma         = atoi(pd.getData( "sigma" ).c_str())/10.0; 

    float scaleFactor = atoi(pd.getData( "scaleFactor" ).c_str())/100.0; 
    if (display)
    {
        cout << "Image height " << height << " width is " << width << endl; 
        cout << "nFeature: " << nfeatures << endl << "nOctaveLayers : " << nOctaveLayers << endl;
        cout << "scaleFactor: " << scaleFactor << endl; 
        cout << "nOctaveLayers: " << nOctaveLayers << endl; 
    }

    auto detector = cv::ORB::create(nfeatures, scaleFactor, nOctaveLayers, 31);
    cout << "Comput the key points " << endl; 
    #pragma omp parallel for
    for (int i = 0; i < source.size(); ++i)
    {
        detector->detectAndCompute(source[i].rgb, cv::Mat(), 
                                    source[i].kp, source[i].desp);
    }
    cout << "Finish Computing the key points" << endl; 
    cout << "building the cloud " << endl; 

    int reference = 0; 
    pointCloud::Ptr cloud = image2PointCloud(source[0],height, width);
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    cv::Ptr<cv::DescriptorMatcher> matcher = 
      cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12,20, 2));
    cout << "Save original cloud" << endl; 
    pcl::io::savePCDFile("Original.pcd", *cloud);
    //pcl::visualization::CloudViewer viewer("viewer");
    //viewer.showCloud( cloud );
    static pcl::VoxelGrid<PointT> voxel;
    double gridsize = 20.02;
    voxel.setLeafSize( gridsize, gridsize, gridsize );


    for (int i = 1; i < 50; i += 2)
    {
        printf("Working on %d reference %d\n",i, reference); 
        FRAME f2 = source[i];
        FRAME f1 = source[reference];
        std::vector<cv::DMatch> goodMatches; 
        
        if (imgDisplay)
        {
            cv::imshow("Frame1", f1.rgb); 
            cv::imshow("Frame2", f2.rgb); 
            cv::waitKey(0); 
        }
        // flann mather
        matcher->match(f1.desp, f2.desp, goodMatches); 
        cout << "good Matches " << goodMatches.size() << endl; 

        // 3D poitns
        std::vector<cv::Point3d> src; 
        std::vector<cv::Point3d> dst; 

        for (size_t i = 0; i<goodMatches.size(); ++i)
        {
             
            cv::Point2d p1 = f1.kp[goodMatches[i].queryIdx].pt;
            cv::Point2d p2 = f2.kp[goodMatches[i].trainIdx].pt;

            cv::Point3d point1; 
            cv::Point3d point2;

            point1.x = f1.depth_x.at<double>(int(p1.y), int(p1.x)); 
            point1.y = f1.depth_y.at<double>(int(p1.y), int(p1.x)); 
            point1.z = f1.depth_z.at<double>(int(p1.y), int(p1.x));

            point2.x = f2.depth_x.at<double>(int(p2.y), int(p2.x)); 
            point2.y = f2.depth_y.at<double>(int(p2.y), int(p2.x)); 
            point2.z = f2.depth_z.at<double>(int(p2.y), int(p2.x));

            src.push_back(point1); 
            dst.push_back(point2);
        }

        int half = src.size() * 0.6;
        double threshold = 5.0; 
        cv::Mat rvec, translationVec, inliers, ratationVector;
        cv::Mat affine = cv::Mat::zeros(3,4,CV_64F);
    
        count = 0; 
        while (count < half)
        {
            threshold += 0.4;
            cv::estimateAffine3D(src, dst, affine,inliers, threshold ,0.98);
            count = 0; 
            for (int i = 0; i < src.size(); ++i)
            {
                if(inliers.at<bool>(0,i) == true)
                {
                    ++count; 
                }
            }
        }
        cout << "Current threshold " << threshold <<endl; 
        


        std::vector<cv::Point3d> srcSVD; 
        std::vector<cv::Point3d> dstSVD; 

        for (int i = 0; i < src.size(); ++i)
        {
            if(inliers.at<bool>(0,i) == true)
            {
                srcSVD.push_back(src[i]); 
                dstSVD.push_back(dst[i]); 
            }
        }
        std::vector<double> t(3); 
        std::vector<double> Rot(3); 
        ResultOfSVD Rt = poseEstimation3D3DReturn(srcSVD, dstSVD, Rot, t);
        T(0,3) = Rt.t_(0,0); 
        T(1,3) = Rt.t_(1,0); 
        T(2,3) = Rt.t_(2,0); 
        T(0,0) = Rt.R_(0,0);
        T(0,1) = Rt.R_(0,1);
        T(0,2) = Rt.R_(0,2);
        T(1,0) = Rt.R_(1,0);
        T(1,1) = Rt.R_(1,1);
        T(1,2) = Rt.R_(1,2);
        T(2,0) = Rt.R_(2,0);
        T(2,1) = Rt.R_(2,1);
        T(2,2) = Rt.R_(2,2);
        pointCloud::Ptr cloud2 = image2PointCloud(source[i],height, width);
        pointCloud::Ptr output (new pointCloud());

        pcl::transformPointCloud( *cloud, *output, T.matrix() );
        *cloud2 += *output;
        cloud = cloud2; 
        voxel.setInputCloud( cloud );
        pointCloud::Ptr tmp( new pointCloud() );
        voxel.filter( *tmp );
        cloud = tmp; 
        
        //viewer.showCloud( cloud );
        
        reference = i; 
        
    }
    pcl::io::savePCDFile("world.pcd", *cloud);

    return 0; 

}