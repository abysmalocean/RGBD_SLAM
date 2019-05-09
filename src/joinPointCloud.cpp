/*
CMSC 591 Slam Project
First Part, VO 
Author: Liang Xu
Data: 04/2019
Contact: liangxuav@gmail.com
*/

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

using std::cout; 
using std::endl;

ResultOfSVD poseEstimation3D3D
(const std::vector<cv::Point3d>& pts1, 
 const std::vector<cv::Point3d>& pts2,
 std::vector<double>& R, 
 std::vector<double>& t)
{
    cv::Point3d p1, p2; 
    int N = pts1.size(); 
    for (int i = 0; i < N; ++i)
    {
        p1 += pts1[i]; 
        p2 += pts2[i]; 
    }
    p1 = cv::Point3d(cv::Vec3d(p1) / N); 
    p2 = cv::Point3d(cv::Vec3d(p2) / N); 
    std::vector<cv::Point3i>     q1 ( N ), q2 ( N ); // remove the center
    for ( int i=0; i<N; i++ )
    {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }
    // compute q1 * q2 ^ t
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; i++)
    {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * 
             Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose(); 
    }
    // SVD on W 
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU|Eigen::ComputeFullV); 
    Eigen::Matrix3d U = svd.matrixU(); 
    Eigen::Matrix3d V = svd.matrixV(); 

    if (U.determinant() * V.determinant() < 0)
	{
        for (int x = 0; x < 3; ++x)
        {
            U(x, 2) *= -1;
        }
	}
    Eigen::Matrix3d R_ = U * (V.transpose()); 

    Eigen::Vector3d t_ = Eigen::Vector3d ( p1.x, p1.y, p1.z ) - 
                         R_ * Eigen::Vector3d ( p2.x, p2.y, p2.z );

    

    cv::Mat mat_r = (cv::Mat_<double>(3, 3) << R_(0, 0), R_(0, 1), R_(0, 2),
             R_(1, 0), R_(1, 1), R_(1, 2),
             R_(2, 0), R_(2, 1), R_(2, 2));
    cv::Mat rotationVector;

    cv::Rodrigues(mat_r,rotationVector);
    cout<<"\n Rotation Vector [Ransac SVD]:\n " << rotationVector * (180.0 / 3.14) << endl; 
    cout <<"\n Translation vector [Ransac SVD]: \n" << t_ << endl; 


    // convert to cv::Mat
    auto rot = Eigen::AngleAxisd(R_).axis();
    R[0] = rot[0]; 
    R[1] = rot[1]; 
    R[2] = rot[2]; 
    t[0] = t_(0,0); 
    t[1] = t_(1,0); 
    t[2] = t_(2,0); 
    ResultOfSVD result; 
    result.rvec = mat_r; 
    result.tvec = (cv::Mat_<double>(3,1) << t_(0,0), t_(1,0), t_(2,0)); 
    result.R_   = R_; 
    result.t_   = t_; 
    //cout << result.rvec << endl; 
    //cout << result.tvec << endl; 
    return result; 

}


int main( int argc, char** argv )
{
    //Read data
    ParameterReader pd;
    int display = atoi( pd.getData( "display" ).c_str() );
    int imgDisplay = atoi( pd.getData( "imageDisplay" ).c_str() );
    int width  =   atoi( pd.getData( "width" ).c_str() );
    int height    =   atoi( pd.getData( "height"   ).c_str() );

    std::string folder = pd.getData("DataFolder"); 
    std::vector<FRAME> source; 

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
			
		}
	}
    count = 0;
    std::sort(paths.begin(), paths.end()); 
    int frame1 = atoi( pd.getData( "frame1" ).c_str() );
    int frame2 = atoi( pd.getData( "frame2" ).c_str() );

    FRAME f1 = readImage(paths[frame1], &pd);
    FRAME f2 = readImage(paths[frame2], &pd);
    std::cout << "working on : \n" << paths[frame1] << "\n" << paths[frame2] << endl;  
    
    if (imgDisplay)
    {
        cv::imshow("Frame1", f1.rgb); 
        cv::imshow("Frame2", f2.rgb); 
        cv::waitKey(0); 
    } 

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
    
    detector->detectAndCompute(f1.rgb, cv::Mat(), f1.kp, f1.desp);

    detector->detectAndCompute(f2.rgb, cv::Mat(), f2.kp, f2.desp); 
    // find matches
    std::vector<cv::DMatch> matches; 
    // flann mather
    cv::Ptr<cv::DescriptorMatcher> matcher = 
        cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12,20, 2));
    cout << "size of f1" << f1.desp.size() << " f2 " << f2.desp.size() << endl;
    matcher->match(f1.desp, f2.desp, matches); 

    std::vector<cv::DMatch> goodMatches; 
    double minDis = 999;
    // get the smallest dist 
    for (size_t i = 0; i < matches.size(); ++i)
    {
        if ( matches[i].distance < minDis )
            minDis = matches[i].distance;
    }
    minDis += 0.000001; 
    //cout<<"min dis = "<<minDis<<endl;
    // get the good matches
    int scaleOfGoodMatch = atoi( pd.getData( "scaleOfGoodMatch" ).c_str() );
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if (matches[i].distance <= scaleOfGoodMatch*minDis)
            goodMatches.push_back( matches[i] );
    }
    // display good matches
    if (display)
    {
        cout<<"Find total "<<matches.size()<<" matches."<<endl;
        cout<<"good matches="<<goodMatches.size()<<endl;
    }

    if (imgDisplay)
    {
        cv::Mat imgMatches;
        cv::drawMatches( f1.rgb, f1.kp, f2.rgb, f2.kp, goodMatches, imgMatches );
        cv::imwrite( "good_matches.png", imgMatches );
        cv::imshow( "good matches", imgMatches );
        cv::waitKey(0); 
    }

    // 3D poitns
    std::vector<cv::Point3d> src ; 
    std::vector<cv::Point3d> dst; 
    // 2D location
    std::vector<cv::Point2d> imagCor; 

    for (size_t i = 0; i<goodMatches.size(); ++i)
    {

        
        cv::Point2d p1 = f1.kp[goodMatches[i].queryIdx].pt;
        cv::Point2d p2 = f2.kp[goodMatches[i].trainIdx].pt;
        
        cv::Point3d point1; 
        cv::Point3d point2;
        //cout << p1.x << " " << p2.x << endl; 
        //cout << p1.y << " " << p2.y << endl; 
        //cout << endl;  

        point1.x = f1.depth_x.at<double>(int(p1.y), int(p1.x)); 
        point1.y = f1.depth_y.at<double>(int(p1.y), int(p1.x)); 
        point1.z = f1.depth_z.at<double>(int(p1.y), int(p1.x));

        point2.x = f2.depth_x.at<double>(int(p2.y), int(p2.x)); 
        point2.y = f2.depth_y.at<double>(int(p2.y), int(p2.x)); 
        point2.z = f2.depth_z.at<double>(int(p2.y), int(p2.x));

        src.push_back(point1); 
        dst.push_back(point2);
    }
    
    cv::Mat rvec, translationVec, inliers, ratationVector;
    cv::Mat affine = cv::Mat::zeros(3,4,CV_64F);
    if (display)
    {
        cout<<"src.size "<<src.size()<<endl;
        cout<<"dst.size "<<dst.size()<<endl;
    }
    
    int half = src.size() * 0.7;
    double threshold = 10.0; 
    
    count = 0; 
    while (count < half)
    {
        threshold += 0.1;
        cv::estimateAffine3D(src, dst,affine,inliers, threshold ,0.99);
        count = 0; 
        for (int i = 0; i < src.size(); ++i)
        {
            if(inliers.at<bool>(0,i) == true)
            {
                ++count; 
            }
        }
    }
    
    std::cout << "Inliners : " << count << " Total : " << src.size() << std::endl;
    std::cout << "thres hold " << threshold << std::endl; 

    int writeImg = atoi( pd.getData( "writeImg" ).c_str() );
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

    if (writeImg)
    {
        cv::Mat imgMatches;
        std::vector<cv::DMatch> goodMatches2;
        for (int i = 0; i < src.size(); ++i)
        {
            if(inliers.at<bool>(0,i) == true)
            {
                goodMatches2.push_back(goodMatches[i]); 
            }
        }
        cv::drawMatches( f1.rgb, f1.kp, f2.rgb, f2.kp, goodMatches2, imgMatches );
        cv::imwrite( "good_matches.png", imgMatches );
    }

    std::vector<double> t(3); 
    std::vector<double> Rot(3); 

    ResultOfSVD Rt = poseEstimation3D3D(srcSVD, dstSVD, Rot, t);
    Eigen::Isometry3d transform_2 = Eigen::Isometry3d::Identity();
    //Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();

    Eigen::AngleAxisd angle(Rt.R_);
    //cout << angle << endl; 
    //T = angle; 
    
    Eigen::Translation<double,3> trans(Rt.t_(0,0), Rt.t_(1,0), Rt.t_(2,0));
    transform_2(0,3) = Rt.t_(0,0); 
    transform_2(1,3) = Rt.t_(1,0); 
    transform_2(2,3) = Rt.t_(2,0); 
    transform_2(0,0) = Rt.R_(0,0);
    transform_2(0,1) = Rt.R_(0,1);
    transform_2(0,2) = Rt.R_(0,2);
    transform_2(1,0) = Rt.R_(1,0);
    transform_2(1,1) = Rt.R_(1,1);
    transform_2(1,2) = Rt.R_(1,2);
    transform_2(2,0) = Rt.R_(2,0);
    transform_2(2,1) = Rt.R_(2,1);
    transform_2(2,2) = Rt.R_(2,2);

    cout << "Rows " <<transform_2.rows() << " Cols " << transform_2.cols() << endl; 
    //cout << "T " << endl; 
    //cout << T << endl; 

    cout<<"converting image to clouds"<<endl;

    pointCloud::Ptr cloud1 = image2PointCloud(f1,height, width);
    pointCloud::Ptr cloud2 = image2PointCloud(f2,height, width);
    cout << "Saving the sub clould " << endl; 
    pcl::io::savePCDFile("clould1.pcd", *cloud1);
    pcl::io::savePCDFile("clould2.pcd", *cloud2);

    // combine cloulds
    pointCloud::Ptr output (new pointCloud());
    cout<<"combining clouds"<<endl;
    cout << transform_2.matrix() << endl; 

    pcl::transformPointCloud( *cloud1, *output, transform_2.matrix());
    
    cout << "Combine result " << endl; 
    *output += *cloud2;

    pcl::io::savePCDFile("result.pcd", *output);


    return 0; 

}

