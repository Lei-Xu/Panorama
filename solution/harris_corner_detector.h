//
// Created by Lei Xu on 2019-01-23.
//

#ifndef SOLUTION_HARRIS_CORNER_DETECTOR_H
#define SOLUTION_HARRIS_CORNER_DETECTOR_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <math.h>

using namespace cv;
using namespace std;

class HarrisCorner {
public:
    void detector(Mat& src, Mat& imgDst, double qualityLevel, vector<cv::Point2f> &points);

    void guassian(int sizex, int sizey, double sigma, Mat &kernel);
};

#endif //SOLUTION_HARRIS_CORNER_DETECTOR_H
