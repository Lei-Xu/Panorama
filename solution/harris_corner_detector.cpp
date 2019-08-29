//
// Created by Lei Xu on 2019-01-28.
//

#include "harris_corner_detector.h"


void HarrisCorner::detector(Mat& src, Mat& imgDst, double qualityLevel, vector<cv::Point2f> &points) {
    Mat Dx, Dy;
    Mat DxDy,Dx2, Dy2;
    Mat dilated;
    Mat localMax;
    Mat cornerMap;
    Mat gray;
    Mat cornerTh;


    if (src.channels() == 3)
    {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    }
    else
    {
        gray = src.clone();
    }
    gray.convertTo(gray, CV_64F);

    //// Sobel operator sx - 1a
    float kernel_data0[9]={ -1, 0, 1,

                            -1, 0, 1,

                            -1, 0, 1};


    Mat kernel3 = Mat(3, 3, CV_32F, kernel_data0);
    filter2D(gray, Dx, gray.depth(), kernel3);

    //// Sobel operator sy - 1a
    float kernel_data1[9]={ -1, -1, -1,

                            0,   0,  0,

                            1,   1,  1};

    Mat kernel4 = Mat(3, 3, CV_32F, kernel_data1);
    filter2D(gray, Dy, gray.depth(), kernel4);

    ////1b
    DxDy = Dx.mul(Dy);
    Dy2 = Dy.mul(Dy);
    Dx2 = Dx.mul(Dx);

    Mat Kernel(3, 3, CV_64F);
    guassian(3, 3, 1 , Kernel);

    filter2D(DxDy, DxDy, DxDy.depth(), Kernel);
    filter2D(Dx2, Dx2, Dx2.depth(), Kernel);
    filter2D(Dy2, Dy2, Dy2.depth(), Kernel);

    Mat cornerStrength(gray.size(), gray.type());
    for (int i = 0; i < gray.rows; i++)
    {
        for (int j = 0; j < gray.cols; j++)
        {
            double det_m = Dx2.at<double>(i,j) * Dy2.at<double>(i,j) - DxDy.at<double>(i,j) * DxDy.at<double>(i,j);
            double trace_m = Dx2.at<double>(i,j) + Dy2.at<double>(i,j);
            cornerStrength.at<double>(i,j) = det_m/trace_m; ////1c
        }
    }

    double maxStrength;
    minMaxLoc(cornerStrength, NULL, &maxStrength, NULL, NULL);


    dilate(cornerStrength, dilated, Mat());

    compare(cornerStrength, dilated, localMax, CMP_EQ);

    double thresh = qualityLevel * maxStrength;
    cornerMap = cornerStrength > thresh; ////1d
    bitwise_and(cornerMap, localMax, cornerMap);

    imgDst = cornerMap.clone();

    for( int y = 0; y < cornerMap.rows; y++ ) {
        const uchar* rowPtr = cornerMap.ptr<uchar>(y);
        for( int x = 0; x < cornerMap.cols; x++ ) {
            if (rowPtr[x]) {
                points.push_back(cv::Point2f(x,y));
            }
        }
    }
}

void HarrisCorner::guassian(int sizex, int sizey, double sigma, Mat &kernel) {
    double pi = M_PI;
    double mins = 0;
    if((sizex % 2 == 0)||(sizey % 2 == 0)){
        cout << "Wrong Number" << endl;
        return;
    }
    double mid1 = floor((sizex - 1) / 2);
    double mid2 = floor((sizey - 1) / 2);
    for(int i = 1; i <= sizex; i++){
        for(int j = 1 ; j <= sizey; j++) {
            double ttt = ((i - mid1 - 1) * (i - mid1 - 1) + (j - mid2 - 1) * (j - mid2 - 1)) / (2 * sigma * sigma);
            double t = exp(- ttt);
            double a = t / (2 * pi * sigma * sigma);
            mins += a;
            kernel.at<double>(i - 1, j - 1) = a;
        }
    }

    for(int i = 0; i < sizex; i++){
        for (int j = 0; j < sizey; j++) {
            kernel.at<double>(i, j) /= mins;
        }
    }
}
