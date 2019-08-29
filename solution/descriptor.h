//
// Created by Lei Xu on 2019-01-27.
//

#ifndef SOLUTION_DESCRIPTOR_H
#define SOLUTION_DESCRIPTOR_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <math.h>

using namespace std;
using namespace cv;

class Descriptor {
public:
    Mat getPatch(Mat &src, KeyPoint keyPoint);

    Mat Gaussian(int ksize, float sigma);

    Mat getDescriptor(Mat &src, KeyPoint keyPoint);

    int theta(Mat &patch_mag, Mat &patch_dir);

    Mat descriptor(Mat &img, vector<KeyPoint> &keypoints);

    Mat gaussian_x(int ksize, float sigma);

    Mat gaussian_y(int ksize, float sigma);

    void guassian(int sizex, int sizey, double sigma, Mat &kernel);
};


#endif //SOLUTION_DESCRIPTOR_H
