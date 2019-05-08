/*
CMSC 591 Slam Project
Utility Folder, used for loading data
Author: Liang Xu
Data: 04/2019
Contact: liangxuav@gmail.com
*/
#ifndef _UTILITY_LIB
#define _UTILITY_LIB

#include <fstream>
#include <vector>
#include <map>
#include <iostream>
#include <string>


#include <opencv2/opencv.hpp>
#include <Eigen/Core>



// PCL lib
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>


typedef pcl::PointXYZI PointT; 
typedef pcl::PointCloud<PointT> pointCloud; 

// FRAME Struct
struct FRAME
{
    int frameID; 
    cv::Mat rgb, depth; // image and depth
    cv::Mat depth_x, depth_y, depth_z; 
    cv::Mat desp;       // descriptor
    std::vector<cv::KeyPoint> kp; // key points
};

struct ResultOfSVD
{
    cv::Mat rvec, tvec;
    Eigen::Matrix3d R_; 
    Eigen::Vector3d t_; 
};

// camera interistic parameters
struct CAMERA_INTRINSIC_PARAMETERS 
{ 
    double cx, cy, fx, fy, scale;
};



// read parameters
class ParameterReader
{
public:
    ParameterReader( std::string filename="./parameters.txt" )
    {
        std::ifstream fin( filename.c_str() );
        if (!fin)
        {
            std::cerr<<"parameter file does not exist."<<std::endl;
            return;
        }
        while(!fin.eof())
        {
            std::string str;
            getline( fin, str );
            if (str[0] == '#')
            {
                // 以‘＃’开头的是注释
                continue;
            }

            int pos = str.find("=");
            if (pos == -1)
                continue;
            std::string key = str.substr( 0, pos );
            std::string value = str.substr( pos+1, str.length() );
            data[key] = value;

            if ( !fin.good() )
                break;
        }
    }
    std::string getData( std::string key )
    {
        std::map<std::string, std::string>::iterator iter = data.find(key);
        if (iter == data.end())
        {
            std::cout<<"Parameter name "<<key<<" not found!"<<std::endl;
            return std::string("NOT_FOUND");
        }
        return iter->second;
    }
public:
    std::map<std::string, std::string> data;
};

FRAME readImage(std::string fileName, ParameterReader *pd, int ID = 0); 

pointCloud::Ptr image2PointCloud( FRAME f , int height, int width);


#endif