#include <../include/utility.h>
#include <string>
#include <iostream>
#include <omp.h>



/*
// FRAME Struct
struct FRAME
{
    int frameID; 
    cv::Mat rgb, depth; // image and depth
    cv::Mat desp;       // descriptor
    cv::Mat depth_x, depth_y, depth_z; 
    vector<cv::KeyPoint> kp; // key points
};
*/

// read SwissRanger SR4000
FRAME readImage(std::string FileName, ParameterReader *pd, int ID)
{
    // std::cout<< "in image read file " << std::endl; 

    int width  =   atoi( pd->getData( "width" ).c_str() );
    int height    =   atoi( pd->getData( "height"   ).c_str() );
    int display = atoi( pd->getData( "display" ).c_str() );
    int imageDisplay = atoi( pd->getData( "imageDisplay" ).c_str() );

    FRAME f;
    //cv::Mat gray  = cv::Mat::zeros(height,width, CV_64F); 
    
    cv::Mat gray    = cv::Mat::zeros(height, width, CV_64F); 
    cv::Mat depthX  = cv::Mat::zeros(height, width, CV_64F); 
    cv::Mat depthY  = cv::Mat::zeros(height, width, CV_64F); 
    cv::Mat depthZ  = cv::Mat::zeros(height, width, CV_64F); 

    int size[3] = {height, width, 3}; 
    cv::Mat depth(3, size , CV_64F, cv::Scalar(0));

    std::ifstream imageFile(FileName); 
    std::string str; 
    std::string tmp; 
    size_t lineCount = 0; 
    double currentVal; 

    while(getline(imageFile, str))
    {
        // 
        // every line is 176 data, for each row
        // get the first depth, channel 2
        ++lineCount;
        if (lineCount > 0 && (lineCount < height + 1) ) 
        {
            //std::cout << lineCount - 1 << std::endl; 
            for (int i = 0; i < width; ++i)
            {
                imageFile >> currentVal; 
                depth.at<double>(lineCount-1,i,2) = currentVal;
                depthZ.at<double>(lineCount - 1, i) = currentVal ; 
                //std::cout << depthZ.at<double>(lineCount - 1, i) << std::endl;
            }
        }

        // get the second depth, channel 0 
        else if(lineCount > (height + 1) && lineCount < (height + 1)* 2)
        {
            //std::cout << lineCount - height - 2 << std::endl; 
            for (int i = 0; i < width; ++i)
            {
                /*
                imageFile >> depthX.at<double>(lineCount - height - 2, i);

                depth.at<double>(lineCount - height - 2, i, 0) = 
                        depthX.at<double>(lineCount - height - 2, i);
                */
               imageFile >> currentVal; 
               depth.at<double>(lineCount - height - 2, i, 0) = currentVal;
               depthX.at<double>(lineCount - height - 2, i)   = currentVal ; 
             //std::cout << depthX.at<double>(lineCount - height - 2, i) << std::endl; 
            }
        }

        // get the third depth, channel 1 
        else if(lineCount > (height + 1) * 2 && lineCount < (height + 1) * 3)
        {
            //std::cout << lineCount - 1- 2 * (height + 1) << std::endl;
            for (int i = 0; i < width; ++i)
            {
                /*
                imageFile >> depthY.at<double>(lineCount - 1- 2 * (height + 1), i) ; 
                depth.at<double>(lineCount - 1- 2 * (height + 1), i, 1) = 
                        depthY.at<double>(lineCount - 1- 2 * (height + 1), i); 
                */
                imageFile >> currentVal; 
                depth.at<double>(lineCount - 1- 2 * (height + 1), i, 1) = currentVal; 
                depthY.at<double>(lineCount - 1- 2 * (height + 1), i) = currentVal ; 
            }
        }

        // get the gray image
        else if (lineCount > (height + 1)* 3 && lineCount < (height + 1) * 4)
        {
            for (int i = 0; i < width; ++i)
            {
                imageFile >> currentVal; 
                gray.at<double>(lineCount - 1 - 3 * (height+1), i, 0) = currentVal;
                //std::cout << gray.at<double>(lineCount - 1 - 3 * (height+1), i, 0) << std::endl; 
                //std::cout << gray.at<double>(lineCount - 3*height, i, 0); 
            }
            //std::cout <<std::endl;
        }
    }  
    

    //cv::normalize(gray, gray, 1.0, 0.0, cv::NORM_L2); 
     
    cv::Mat means, stddev;
	cv::meanStdDev(gray, means, stddev);
    if(display)
    {
        printf("mean: %.2f, stddev: %.2f\n", means.at<double>(0), stddev.at<double>(0));
    }
    //gray.convertTo(gray, CV_32F);
    //cv::medianBlur(gray, gray, 5);
    //cv::GaussianBlur(gray, gray, cv::Size(3,3),1.0,1.0,4);
    // filter out the extream large value, this value is error for some of the reason
    #pragma omp parallel for
    for (int j = 0; j < height; ++j)
    {
        for (int i = 0; i < width; ++i)
        {
            if (gray.at<double>(j, i, 0) > means.at<double>(0) + 4 * stddev.at<double>(0)) 
            {
                gray.at<double>(j, i, 0) =  means.at<double>(0) + 4 * stddev.at<double>(0); 
            }
            
        }
    }

    cv::normalize(gray, gray, 1.0, 0.0, cv::NORM_MINMAX);
    gray *= 255.0; 
    gray.convertTo(gray, CV_8UC1);

    // cv::imshow("gray img", gray);
    if (false)
    {
        std::cout << FileName << std::endl; 
        cv::imshow("gray img", gray);

        depthZ *= 255.0; 
        cv::normalize(depthZ, depthZ, 1.0, 0.0, cv::NORM_MINMAX);
        depthZ *= 255.0; 
        depthZ.convertTo(depthZ, CV_8UC1);
        cv::imshow("DepthZ", depthZ);

        cv::normalize(depthX, depthX, 1.0, 0.0, cv::NORM_MINMAX);
        depthX *= 255.0; 
        depthX.convertTo(depthX, CV_8UC1);
        cv::imshow("DepthX", depthX);
        // depthX is Z; 
        cv::normalize(depthY, depthY, 1.0, 0.0, cv::NORM_MINMAX);
        depthY *= 255.0; 
        depthY.convertTo(depthY, CV_8UC1);
        cv::imshow("DepthY", depthY);
        cv::waitKey(0); 
    }

    
    cv::GaussianBlur(depthX,depthX,cv::Size(5,5),1);
    cv::GaussianBlur(depthY,depthY,cv::Size(5,5),1);
    cv::GaussianBlur(depthZ,depthZ,cv::Size(5,5),1);
    //cv::GaussianBlur(gray,gray,cv::Size(3,3),1);

    //cv::equalizeHist(gray,gray); 
    f.rgb = gray.clone(); 
    f.depth = depth.clone(); 

    f.depth_x = (depthX.clone()) * 1000.0; 
    f.depth_y = (depthY.clone()) * 1000.0; 
    f.depth_z = (depthZ.clone()) * 1000.0; 
    return f; 
}


pointCloud::Ptr image2PointCloud( FRAME f , int height, int width)
{

    pointCloud::Ptr cloud(new pointCloud); 
    // interative the point clould
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            PointT p; 
            p.z = f.depth_z.at<double>(i,j); 
            p.x = f.depth_x.at<double>(i,j); 
            p.y = f.depth_y.at<double>(i,j);

            p.intensity = f.rgb.at<double>(i,j); 
            cloud->points.push_back(p); 
        }
    }

    cloud->height = 1; 
    cloud->width = cloud->points.size(); 
    cloud->is_dense = false; 
    return cloud; 
    
}


std::vector<cv::DMatch> 
findGoodMatch(cv::Ptr<cv::FlannBasedMatcher> matcher, FRAME& f1, FRAME& f2 )
{
    std::vector<cv::DMatch> matches;
    matcher->match( f1.desp, f2.desp , matches );

     //-- Quick calculation of max and min distances between keypoints
    double max_dist = 0; double min_dist = 100;
    for( int i = 0; i < f1.desp.rows; i++ )
    { double dist = matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    }

    std::vector< cv::DMatch > goodMatches;

      for( int i = 0; i < f1.desp.rows; i++ )
      { if( matches[i].distance <= std::max(15*min_dist, 0.02) )
        { goodMatches.push_back( matches[i]); }
      }
      return goodMatches; 
}


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
    //cout<<"\n Rotation Vector [Ransac SVD]:\n " << rotationVector * (180.0 / 3.14) << endl; 
    //cout <<"\n Translation vector [Ransac SVD]: \n" << t_ << endl; 


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

ResultOfSVD poseEstimation3D3DReturn
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
    //std::cout<<"\n Rotation Vector [Ransac SVD]:\n " << rotationVector * (180.0 / 3.14) << std::endl; 
    //std::cout <<"\n Translation vector [Ransac SVD]: \n" << t_ << std::endl; 


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