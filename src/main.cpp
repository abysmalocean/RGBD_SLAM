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
    

}

int main( int argc, char** argv )
{
    std::cout<<"Hello 591!"<<std::endl;
    
    //Read data
    ParameterReader pd;
    int display = atoi( pd.getData( "display" ).c_str() );
    int imgDisplay = atoi( pd.getData( "imageDisplay" ).c_str() );
    int width  =   atoi( pd.getData( "width" ).c_str() );
    int height    =   atoi( pd.getData( "height"   ).c_str() );

    std::string filePath1 = pd.getData("testImage"); 

    FRAME f1 = readImage(filePath1, &pd);
    std::cout << "working on : \n" << filePath1 << "\n" ; 
    

    if (imgDisplay)
    {
        cv::imshow("Frame1", f1.rgb); 
        cv::waitKey(0); 
    }    
    
    
    
    return 0; 

}