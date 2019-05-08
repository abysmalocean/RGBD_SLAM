/*
CMSC 591 Slam Project
First Part, VO 
Author: Liang Xu
Data: 04/2019
Contact: liangxuav@gmail.com
*/

#include <iostream>
#include <../include/utility.h>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <dirent.h>
#include <omp.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

using std::cout; 
using std::endl;


void poseEstimation3D3D
(const std::vector<cv::Point3d>& pts1, 
 const std::vector<cv::Point3d>& pts2,
 std::vector<double> &R, std::vector<double>& t)
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
    rotationVector *= (180.0 / 3.14);  
    
    R[0] = rotationVector.at<double>(0); 
    R[1] = rotationVector.at<double>(1); 
    R[2] = rotationVector.at<double>(2); 

    

    t[0] = t_(0,0); 
    t[1] = t_(1,0); 
    t[2] = t_(2,0); 
    

}


int main( int argc, char** argv )
{
    std::cout<<"Hello 591! First part of the project!!"<<std::endl;
    
    //Read data
    ParameterReader pd;
    int display = atoi( pd.getData( "display" ).c_str() );
    int imgDisplay = atoi( pd.getData( "imageDisplay" ).c_str() );
    int width  =   atoi( pd.getData( "width" ).c_str() );
    int height    =   atoi( pd.getData( "height"   ).c_str() );
    std::string filePath1 = pd.getData("testImage1"); 
    std::string filePath2 = pd.getData("testImage2"); 

    FRAME f1 = readImage(filePath1, &pd);
    FRAME f2 = readImage(filePath2, &pd);

    int nfeatures = atoi( pd.getData( "nfeatures" ).c_str() );
    int nOctaveLayers =  atoi( pd.getData( "nOctaveLayers" ).c_str() );
    double contrastThreshold = atoi(pd.getData( "contrastThreshold" ).c_str())/100.0; 
    double edgeThreshold  = atoi(pd.getData( "edgeThreshold" ).c_str()) * 1.0; 
    double sigma         = atoi(pd.getData( "sigma" ).c_str())/10.0; 
    float scaleFactor = atoi(pd.getData( "scaleFactor" ).c_str())/100.0; 
    auto detector = cv::ORB::create(nfeatures, scaleFactor, nOctaveLayers, 31);
    
    
    std::vector<FRAME> source; 
    std::vector<FRAME> distination;

    std::string folder1 = pd.getData("folder1"); 
    std::string folder2 = pd.getData("folder2");
    cout << "Working on Folder :" << folder1 << endl; 
    cout << "Working on Folder :" << folder2 << endl; 

    size_t count = 0; 

    if (auto dir = opendir(folder1.c_str())) 
    {
        while (auto f = readdir(dir)) 
        {
            if (!f->d_name || f->d_name[0] == '.')
                continue; // Skip everything that starts with a dot
            std::string filePath = folder1 + f->d_name; 
            if(display) {printf("File: %s | num [%zu]\n", f->d_name, count);}
            FRAME frame = readImage(filePath, &pd);
            frame.frameID = count++; 
            source.push_back(frame); 
        }
        closedir(dir);
    }
    printf("Source finish \n"); 
    if (auto dir = opendir(folder2.c_str())) 
    {
        while (auto f = readdir(dir)) 
        {
            if (!f->d_name || f->d_name[0] == '.')
                continue; // Skip everything that starts with a dot
            std::string filePath = folder2 + f->d_name; 
            if(display) {printf("File: %s | num [%zu]\n", f->d_name, count);}
            FRAME frame = readImage(filePath, &pd);
            frame.frameID = count++; 
            distination.push_back(frame); 
        }
        closedir(dir);
    }
    printf("finsih loading the data\n"); 

    #pragma omp parallel for
    for (int i = 0; i < source.size(); ++i)
    {
        detector->detectAndCompute(source[i].rgb, cv::Mat(), 
                                    source[i].kp, source[i].desp);
    }

    #pragma omp parallel for
    for (int i = 0; i < distination.size(); ++i)
    {
        detector->detectAndCompute(distination[i].rgb, cv::Mat(), 
                                    distination[i].kp, distination[i].desp);
    }
    printf("Source finish calcaulte the descriptor\n"); 
    cv::Mat R0 = cv::Mat::zeros(source.size(), distination.size(), CV_64F);  
    cv::Mat R1 = cv::Mat::zeros(source.size(), distination.size(), CV_64F);  
    cv::Mat R2 = cv::Mat::zeros(source.size(), distination.size(), CV_64F);  
    cv::Mat T0 = cv::Mat::zeros(source.size(), distination.size(), CV_64F);  
    cv::Mat T1 = cv::Mat::zeros(source.size(), distination.size(), CV_64F);  
    cv::Mat T2 = cv::Mat::zeros(source.size(), distination.size(), CV_64F);  
    int scaleOfGoodMatch = atoi( pd.getData( "scaleOfGoodMatch" ).c_str() );


    //#pragma omp parallel for
    for (int sourceIndex = 0; sourceIndex < source.size(); ++sourceIndex)
    {
            if (sourceIndex%1 == 0)
            {
                printf("working on [%d]\n", sourceIndex ); 
            }
            f1 = source[sourceIndex]; 
        
        for (int distIndex = 0; distIndex < distination.size(); ++distIndex)
        {
            f2 = distination[distIndex]; 
            std::vector<cv::DMatch> matches; 
            // flann mather
            cv::Ptr<cv::DescriptorMatcher> matcher = 
            cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12,20, 2));
            matcher->match(source[sourceIndex].desp, distination[distIndex].desp, matches);
            //std::cout << matches.size() << std::endl; 

            std::vector<cv::DMatch> goodMatches; 

            double minDis = 999;
            // get the smallest dist 
            for (size_t i = 0; i < matches.size(); ++i)
            {
                if ( matches[i].distance < minDis )
                    minDis = matches[i].distance;
            }
            minDis += 0.000001; 
            
            for ( size_t i=0; i<matches.size(); i++ )
            {
                if (matches[i].distance <= scaleOfGoodMatch*minDis)
                    goodMatches.push_back( matches[i] );
            }
            // 3D poitns
            std::vector<cv::Point3d> src; 
            std::vector<cv::Point3d> dst; 
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
            //std::cout << "size is " <<src.size() << std::endl; 

            int half = src.size() * 0.7;
            double threshold = 10.0; 
            int count = 0; 
            while (count < half && threshold < 20.0)
            {
                threshold += 0.5;
                cv::estimateAffine3D(src, dst,affine,inliers, threshold ,0.9999);
                count = 0; 
                for (int i = 0; i < src.size(); ++i)
                {
                    if(inliers.at<bool>(0,i) == true)
                    {
                        ++count; 
                    }
                }
            }

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
            poseEstimation3D3D(srcSVD, dstSVD, Rot, t);
            
            cv::Mat ratationMatrix = affine(cv::Rect(0,0,3,3));
            translationVec = affine(cv::Rect(3,0,1,3));
            cv::Rodrigues(ratationMatrix,ratationVector);
            ratationVector = ratationVector * (180.0 / 3.14); 

            R0.at<double>(sourceIndex, distIndex) = Rot[0]; 
            R1.at<double>(sourceIndex, distIndex) = Rot[1]; 
            R2.at<double>(sourceIndex, distIndex) = Rot[2]; 
            
            T0.at<double>(sourceIndex, distIndex) = t[0];
            T1.at<double>(sourceIndex, distIndex) = t[1]; 
            T2.at<double>(sourceIndex, distIndex) = t[2]; 

            if (display)
            {
                cout<<"R="<<ratationVector<<endl;
                cout<<"t="<<translationVec<<endl;
            }

        }
        
    }
    cv::Mat means, stddev;
	cv::meanStdDev(R0, means, stddev);
    printf("R0: mean: %.2f, stddev: %.2f\n", means.at<double>(0), stddev.at<double>(0));
	cv::meanStdDev(R1, means, stddev);
    printf("R1: mean: %.2f, stddev: %.2f\n", means.at<double>(0), stddev.at<double>(0));
	cv::meanStdDev(R2, means, stddev);
    printf("R2: mean: %.2f, stddev: %.2f\n", means.at<double>(0), stddev.at<double>(0));
    cv::meanStdDev(T0, means, stddev);
    printf("T0: mean: %.2f, stddev: %.2f\n", means.at<double>(0), stddev.at<double>(0));
	cv::meanStdDev(T1, means, stddev);
    printf("T1: mean: %.2f, stddev: %.2f\n", means.at<double>(0), stddev.at<double>(0));
	cv::meanStdDev(T2, means, stddev);
    printf("T2: mean: %.2f, stddev: %.2f\n", means.at<double>(0), stddev.at<double>(0));


    return 0; 

}