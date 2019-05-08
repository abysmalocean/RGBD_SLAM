#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <../include/utility.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

//typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointXYZI PointT; 
typedef pcl::PointCloud<PointT> pointClould; 


int main( int argc, char** argv )
{
    std::cout<<"Display the Point Cloud"<<std::endl; 
    CAMERA_INTRINSIC_PARAMETERS camPara; 
    camPara.cx = 87.5099; 
    camPara.cy = 71.4768; 
    camPara.fx = 222.855; 
    camPara.fy = 225.779; 
    camPara.scale = 1000.0; 

    ParameterReader pd;
    int width  =   atoi( pd.getData( "width" ).c_str() );
    int height    =   atoi( pd.getData( "height"   ).c_str() );

    std::string filePath1 = pd.getData("testImage1");

    FRAME f1 = readImage(filePath1, &pd);

    pointClould::Ptr cloud(new pointClould); 
    // interative the point clould
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            PointT p; 
            p.z = f1.depth_z.at<double>(i,j) * camPara.scale; 
            p.x = f1.depth_x.at<double>(i,j) * camPara.scale; 
            p.y = f1.depth_y.at<double>(i,j) * camPara.scale;

            p.intensity = f1.rgb.at<double>(i,j); 
            cloud->points.push_back(p); 
        }
    }

    cloud->height = 1; 
    cloud->width = cloud->points.size(); 
    cloud->is_dense = false; 
    pcl::io::savePCDFile("./pointclould.pcd", *cloud);
    cloud->points.clear(); 
    std::cout << "Points are saved, clould are generated" << std::endl; 





    return 0; 
}