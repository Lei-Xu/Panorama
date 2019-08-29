#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "vector"
#include "harris_corner_detector.h"
#include "descriptor.h"
#include "matcher.h"


using namespace cv;
using namespace std;

int main() {
//    Mat image_1 = imread("project_images/Rainier1.png", 1);
//    Mat image_2 = imread("project_images/Rainier2.png", 1);
//    Mat image_3 = imread("project_images/Rainier3.png", 1);
//    Mat image_4 = imread("project_images/Rainier4.png", 1);
//    Mat image_5 = imread("project_images/Rainier5.png", 1);
//    Mat image_6 = imread("project_images/Rainier6.png", 1);
    Mat image_1 = imread("project_images/lb1.jpeg", 1);
    Mat image_2 = imread("project_images/lb2.jpeg", 1);
//    Mat image_3 = imread("project_images/lb3.jpeg", 1);

//    Mat image_4 = imread("project_images/streetview4.png", 1);
//    Mat image_5 = imread("project_images/streetview5.png", 1);
//    Mat image_6 = imread("project_images/streetview6.png", 1);
//    Mat image_56 = imread("Stitch56.png", 1);
//    Mat image_564 = imread("Stitch564.png", 1);
//    Mat image_5642 = imread("Stitch5642.png", 1);
//    Mat image_56423 = imread("Stitch56423.png", 1);

//    Mat image_boxes = imread("project_images/Boxes.png", 1);

//    Mat imgs_1 = imread("image_set/scale1.png", 1);
//    Mat imgs_2 = imread("image_set/scale2.png", 1);


    // Downsampling
//    Mat tempImage_1, tempImage_11, tempImage_2, tempImage_22;
//    pyrDown(image_3, tempImage_1);
//    pyrDown(tempImage_1, image_3);
//    imwrite("project_images/lb3.jpeg", image_3);
//    pyrDown(image_2, tempImage_2);
//    pyrDown(tempImage_2, image_2);
//    imwrite("project_images/lb2.jpeg", image_2);
//
//    return 0;

    HarrisCorner harrisCorner;
    Descriptor descriptor;
    Mat dst1;
    Mat dst2;
    Mat dst3;
//    Mat dst4;
//    Mat dst5;
//    Mat dst6;
    Mat dst56;
//    Mat dst564;
//    Mat dst5642;
//    Mat dst56423;
//    Mat dst_boxes;

    ////sv-0.05 lb-0.3
    double threshold_1 = 0.5;
    double threshold_2 = 0.5;
    double threshold_3 = 0.5;
    double threshold_4 = 0.15;
    double threshold_5 = 0.2;
    double threshold_6 = 0.15;
    double threshold_56 = 0.4;
    double threshold_564 = 0.15;
    double threshold_5642 = 0.15;
    double threshold_56423 = 0.15;
    double threshold_boxes = 0.15;
    vector<cv::Point2f> featurePts1;
    vector<cv::Point2f> featurePts2;
    vector<cv::Point2f> featurePts3;
    vector<cv::Point2f> featurePts4;
    vector<cv::Point2f> featurePts5;
    vector<cv::Point2f> featurePts6;
    vector<cv::Point2f> featurePts56;
    vector<cv::Point2f> featurePts564;
    vector<cv::Point2f> featurePts5642;
    vector<cv::Point2f> featurePts56423;
    vector<cv::Point2f> featurePts_boxes;
    vector<KeyPoint> keyPoint1;
    vector<KeyPoint> keyPoint2;
    vector<KeyPoint> keyPoint3;
    vector<KeyPoint> keyPoint4;
    vector<KeyPoint> keyPoint5;
    vector<KeyPoint> keyPoint6;
    vector<KeyPoint> keyPoint56;
    vector<KeyPoint> keyPoint564;
    vector<KeyPoint> keyPoint5642;
    vector<KeyPoint> keyPoint56423;
    vector<KeyPoint> keyPoint_boxes;

    harrisCorner.detector(image_1, dst1, threshold_1, featurePts1);
    harrisCorner.detector(image_2, dst2, threshold_2, featurePts2);
//    harrisCorner.detector(image_3, dst3, threshold_3, featurePts3);
//    harrisCorner.detector(image_4, dst4, threshold_4, featurePts4);
//    harrisCorner.detector(image_5, dst5, threshold_5, featurePts5);
//    harrisCorner.detector(image_6, dst6, threshold_6, featurePts6);
//    harrisCorner.detector(image_56, dst56, threshold_56, featurePts56);
//    harrisCorner.detector(image_564, dst564, threshold_564, featurePts564);
//    harrisCorner.detector(image_5642, dst5642, threshold_5642, featurePts5642);
//    harrisCorner.detector(image_56423, dst56423, threshold_56423, featurePts56423);
//    harrisCorner.detector(image_boxes, dst_boxes, threshold_boxes, featurePts_boxes);


    KeyPoint::convert(featurePts1, keyPoint1);
    KeyPoint::convert(featurePts2, keyPoint2);

//    KeyPoint::convert(featurePts3, keyPoint3);
//    KeyPoint::convert(featurePts4, keyPoint4);
//    KeyPoint::convert(featurePts5, keyPoint5);
//    KeyPoint::convert(featurePts6, keyPoint6);
//    KeyPoint::convert(featurePts56, keyPoint56);
    cout<<"image1: "<<keyPoint1.size()<<endl;
    cout<<"image2: "<<keyPoint2.size()<<endl;
//    KeyPoint::convert(featurePts564, keyPoint564);
//    KeyPoint::convert(featurePts5642, keyPoint5642);
//    KeyPoint::convert(featurePts56423, keyPoint56423);
//    KeyPoint::convert(featurePts_boxes, keyPoint_boxes);

    Mat descriptor_1 = descriptor.descriptor(image_1, keyPoint1);
//    Mat image_matching1;
//    drawKeypoints(image_1, keyPoint1, image_matching1);
//    imshow("Interest Points 1", image_matching1);
//    imwrite("1b.png", image_matching1);

    Mat descriptor_2 = descriptor.descriptor(image_2, keyPoint2);

//    Mat image_matching2;
//    drawKeypoints(image_2, keyPoint2, image_matching2);
//    imshow("Interest Points 2", image_matching2);
//    imwrite("1c.png", image_matching2);

//    Mat descriptor_boxes = descriptor.descriptor(image_boxes, keyPoint_boxes);
//    Mat image_matching_boxes;
//    drawKeypoints(image_boxes, keyPoint_boxes, image_matching_boxes);
//    imshow("Interest Points Boxes", image_matching_boxes);
//    imwrite("1a.png", image_matching_boxes);

//    Mat descriptor_3 = descriptor.descriptor(image_3, keyPoint3);

//    Mat descriptor_4 = descriptor.descriptor(image_4, keyPoint4);
//
//    Mat descriptor_5 = descriptor.descriptor(image_5, keyPoint5);
//
//    Mat descriptor_6 = descriptor.descriptor(image_6, keyPoint6);

//    Mat descriptor_56 = descriptor.descriptor(image_56, keyPoint56);

//    Mat descriptor_564 = descriptor.descriptor(image_564, keyPoint564);
//
//    Mat descriptor_5642 = descriptor.descriptor(image_5642, keyPoint5642);

//    Mat descriptor_56423 = descriptor.descriptor(image_56423, keyPoint56423);

    Matcher matcher(descriptor_2, descriptor_1);
    Mat stitch1;


    matcher.run(image_2, image_1, keyPoint2, keyPoint1, stitch1);

    imshow("Stitched", stitch1);
    imwrite("Stitch56.png", stitch1);

    waitKey(0);
    return 0;
}