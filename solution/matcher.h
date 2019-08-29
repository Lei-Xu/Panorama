//
// Created by Lei Xu on 2019-01-28.
//

#ifndef SOLUTION_MATCHER_H
#define SOLUTION_MATCHER_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/core/types_c.h"
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include <random>

using namespace cv;
using namespace std;


class Matcher {
private:
    float ssd;
    Mat src1;
    Mat Src2;

public:
    Matcher(const Mat &src1, const Mat &Src2);

    void run(Mat& image1, Mat& image2, vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, Mat& stitchs);

    float Ssd(int r1, int r2);

    void ratio(vector<DMatch> &matches);

    const Mat &getSrc1() const;

    const Mat &getSrc2() const;

    void
    computeInlierCount(Mat &H, vector<DMatch> matches, int &numMatches, float inlierThrehold, vector<KeyPoint> keypoint1,
                       vector<KeyPoint> keypoint2);

    void project(float x1, float y1, Mat &H, float &x2, float &y2);

    void RANAC(vector<DMatch> matches, int &numMatches, int numIterations, vector<KeyPoint> keypoint1,
               vector<KeyPoint> keypoint2, Mat &hom, Mat &homlnv, Mat &image1Display, Mat &image2Display);

    void findInlier(Mat &H, vector<DMatch> &matches, vector<DMatch> &matches2, float inlierThrehold,
                       vector<KeyPoint> &keypoint1, vector<KeyPoint> &keypoint2);

    void stitch(Mat &image1, Mat &image2, Mat &hom, Mat &homInv, Mat &stitchedImage, Mat& stitch);
};

#endif //SOLUTION_MATCHER_H
