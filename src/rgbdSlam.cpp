#include <iostream>
#include <../include/utility.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <dirent.h>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <opencv2/xfeatures2d/nonfree.hpp>

#include <omp.h>

//g2o的头文件
#include <g2o/types/slam3d/types_slam3d.h> //顶点类型
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
//#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>


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
    int nOctaveLayers =  atoi( pd.getData( "nOctaveLayers" ).c_str() );
    double contrastThreshold = atoi(pd.getData( "contrastThreshold" ).c_str())/1000.0; 
    cv::Ptr<cv::Feature2D> detector = 
            cv::xfeatures2d::SIFT::create(0,nOctaveLayers, contrastThreshold);

    #pragma omp parallel for
    for (int i = 0; i < source.size(); ++i)
    {
        detector->detectAndCompute(source[i].rgb, cv::Mat(), 
                                    source[i].kp, source[i].desp);
    }
    cout << "Finish Computing the key points" << endl; 

    typedef g2o::BlockSolver_6_3 SlamBlockSolver; 
    typedef g2o::g2o::LinearSolverCholmod< SlamBlockSolver::PoseMatrixType > SlamLinearSolver;
    

   
    return 0; 
}

