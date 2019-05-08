/*
CMSC 591 Slam Project
First Part, VO 
Author: Liang Xu
Data: 04/2019
Contact: liangxuav@gmail.com
*/


#include <iostream>
#include <../include/utility.h>
//#include <opencv2/xfeatures2d/nonfree.hpp>
//#include <opencv2/opencv.hpp>
#include <vector>

#include <Eigen/Core>
//#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/Dense>

using std::cout; 
using std::endl;

int main( int argc, char** argv )
{
    std::cout<<"Hello 591!"<<std::endl;
    
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
    std::cout << "working on : \n" << filePath1 << "\n" << filePath2 << endl;  

    if (imgDisplay)
    {
        cv::imshow("Frame1", f1.rgb); 
        cv::imshow("Frame2", f2.rgb); 
        cv::waitKey(0); 
    }
    //int height = f1.depth_x.size().height; 
    //int width  = f1.depth_x.size().width ; 
    const int len = f1.depth_x.size().height * f1.depth_x.size().width; 

    //Eigen::Matrix<double, len, 2> A = Eigen::Matrix<double, len, 2>::Zero(len, 2);
    Eigen::MatrixXd A(len*2,2); 
    Eigen::VectorXd b(len*2);

    cout << len << endl; 

    for (int i = 0; i < height; ++i)
    {
        //cout << i << endl; 
        for (int j = 0; j < width; ++j)
        {
            //cout << i * width + j  << endl; 
            A(i * width + j, 0) = f1.depth_x.at<double>(i,j)/f1.depth_z.at<double>(i,j); 
            A(i * width + j, 1) = 1.0; 
            b(i * width + j   ) = -j;
            
        }
    }

    for (int i = 0; i < height; ++i)
    {
        //cout << i << endl; 
        for (int j = 0; j < width; ++j)
        {
            //cout << i * width + j  << endl; 
            A(i * width + j+len, 0) = f2.depth_x.at<double>(i,j)/f2.depth_z.at<double>(i,j); 
            A(i * width + j+len, 1) = 1.0; 
            b(i * width + j+len   ) = -j;
            
        }
    }


    cout << "Intrinc parameters are fx , cx" << endl; 
    cout << A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b); 

    for (int i = 0; i < height; ++i)
    {
        //cout << i << endl; 
        for (int j = 0; j < width; ++j)
        {
            //cout << i * width + j  << endl; 
            A(i * width + j, 0) = f1.depth_y.at<double>(i,j)/f1.depth_z.at<double>(i,j); 
            A(i * width + j, 1) = 1.0; 
            b(i * width + j   ) = -i;
            
        }
    }

    for (int i = 0; i < height; ++i)
    {
        //cout << i << endl; 
        for (int j = 0; j < width; ++j)
        {
            //cout << i * width + j  << endl; 
            A(i * width + j + len, 0) = f2.depth_y.at<double>(i,j)/f2.depth_z.at<double>(i,j); 
            A(i * width + j + len, 1) = 1.0; 
            b(i * width + j + len   ) = -i;
            
        }
    }
    cout << endl<< "<Intrinc parameters are fy , cy" << endl; 
    cout << A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b); 


    return 0; 
}