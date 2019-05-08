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
    for (int i = 0; i < paths.size(); ++i)
    {
        FRAME frame = readImage(paths[i], &pd);
        frame.frameID = count++; 
        source.push_back(frame); 

    }
    printf("Source finish \n");
    
    if (imgDisplay)
    {
        for (int i = 0; i < source.size(); ++i)
        {
            cv::imshow("Frame1", source[i].rgb); 
            cv::waitKey(0); 
        }
        
    } 
    
    return 0; 

}