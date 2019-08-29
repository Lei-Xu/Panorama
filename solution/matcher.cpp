//
// Created by Lei Xu on 2019-01-28.
//

#include "matcher.h"

Matcher::Matcher(const Mat &src1, const Mat &Src2) : src1(src1), Src2(Src2) {
}

////2b
float Matcher::Ssd(int r1, int r2) {
    ssd = 0;
    for(int i = 0; i < getSrc1().cols; i++)
    {
        ssd +=  (getSrc1().at<float>(r1,i) - getSrc2().at<float>(r2,i)) *
                (getSrc1().at<float>(r1,i) - getSrc2().at<float>(r2,i));
    }
    return ssd;
}

void Matcher::ratio(vector<DMatch> &matches) {
    int minkey = 0;
    float threshold = 0.6;
    for(int row1 = 0; row1 < getSrc1().rows;row1++){
        float distsq2= 10;
        float distsq1= 10;
        minkey= 0 ;
        for(int row2 = 0 ;row2 < getSrc2().rows;row2++){
            float SSD = Ssd(row1, row2);
            if(SSD < distsq1){
                distsq2 = distsq1;
                distsq1 = SSD;
                minkey = row2;
            }
        }
        if((distsq1 / distsq2) < threshold){
            DMatch bestPair(row1 , minkey , distsq1);
            matches.push_back(bestPair);
        }
    }
}

//match image by using ratio test
void Matcher::run(Mat& image1, Mat& image2, vector<KeyPoint>& keypoint1, vector<KeyPoint>& keypoint2, Mat& stitchs) {
    vector< DMatch > matches;

    ratio(matches);

    cout<<matches.size()<<endl;

    Mat image_matches_1;

    ////2d
//    drawMatches( image1, keypoint1, image2, keypoint2, matches, image_matches_1, Scalar::all(-1), Scalar::all(-1),
//                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
//    imshow("first match", image_matches_1);
//    imwrite("2.png", image_matches_1);

    Mat img_matches;

    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    int num = 10;

    Mat h;
    Mat hinv;

    Mat stitchedImage;


    RANAC(matches, num, 2000, keypoint1, keypoint2, h, hinv, image1, image2);
    stitch(image1, image2, h, hinv, stitchedImage, stitchs);


}

const Mat &Matcher::getSrc1() const {
    return src1;
}


const Mat &Matcher::getSrc2() const {
    return Src2;
}

////3a
//一张图中1个点，知道H3*3，求另一张图上的点
void Matcher::project(float x1, float y1, Mat &H, float& x2, float& y2) {
    double w = H.at<double>(2, 0) * x1 + H.at<double>(2, 1) * y1 +  H.at<double>(2, 2);
    x2 = float((H.at<double>(0, 0) * x1 + H.at<double>(0, 1) * y1 +  H.at<double>(0, 2)) / w);
    y2 = float((H.at<double>(1, 0) * x1 + H.at<double>(1, 1) * y1 +  H.at<double>(1, 2)) / w);
}

////3b
void Matcher::computeInlierCount(Mat &H, vector<DMatch> matches, int& numMatches, float inlierThrehold, vector<KeyPoint> keypoint1, vector<KeyPoint> keypoint2) {
    float x, y;
    for(int i = 0; i < matches.size(); i++) {

        if (H.rows == 0 || H.type() == 0) {
            continue;
        }
        //知道左边图的点，H，找出投影到右边的位置
        project(keypoint1[matches[i].queryIdx].pt.x, keypoint1[matches[i].queryIdx].pt.y, H, x, y);

        //计算投影过去的点与应该对应的点的距离
        float rx = (keypoint2[matches[i].trainIdx].pt.x - x) * (keypoint2[matches[i].trainIdx].pt.x - x);
        float ry = (keypoint2[matches[i].trainIdx].pt.y - y) * (keypoint2[matches[i].trainIdx].pt.y - y);
        float distance = sqrt(rx + ry);

        if(distance < inlierThrehold) {
            //若距离小于threshold则inliner的数量加1
            numMatches++;
        }

    }

}


////3c
void Matcher::RANAC(vector<DMatch> matches, int &numMatches, int numIterations, vector<KeyPoint> keypoint1,
                    vector<KeyPoint> keypoint2, Mat &hom, Mat &homlnv, Mat &image1Display,
                    Mat &image2Display) {
    int max_num = 0;
    RNG rng;

    //循环用户设定的次数
    for(int i = 0; i < numIterations; i++) {
        Mat H;

        vector< DMatch > matches2;
        vector<KeyPoint> keyPoints1, keyPoints2;
        vector<Point2f> obj;
        vector<Point2f> scene;

        int m = 4;

        //do random shuffle. 找四对点
        int idxArr[] = {-1,-1,-1,-1};
        for (int i = 0; i < 4; i++) {
            int temp = rng.uniform(0, int(matches.size()));
            for (int j = 0; j < 4; j++) {
                if (temp == idxArr[j]) {
                    i--;
                    break;
                }
            }
            idxArr[i] = temp;
        }


        //将找到的4对点的坐标放入obj和scene， query左边的图，train为右边的图
        for (int j = 0; j < m; j++) {
            keyPoints1.push_back( keypoint1[matches[idxArr[j]].queryIdx]);
            keyPoints2.push_back(keypoint2[matches[idxArr[j]].trainIdx]);

            obj.push_back( keypoint1[matches[idxArr[j]].queryIdx ].pt );
            scene.push_back( keypoint2[matches[idxArr[j]].trainIdx ].pt );

            DMatch bestPair(j , j , 0);
            matches2.push_back(bestPair);

        }
        //用这四对点计算出H
        H = findHomography( obj, scene, 0 );
        int num = 0;
        //根据这个H看能找出几个inliner，num即为找出inliner的数量
        computeInlierCount(H, matches, num, 2, keypoint1, keypoint2);
        std::vector<Point2f> scene_corners(4);

        //记下能找出最多inlier的H，和inlier的数量
        if(num > max_num) {
            max_num = num;
            hom = H;
        }
    }
    //用上面找到最好的H，找出所有的inliner
    vector< DMatch > good_matches;
    findInlier(hom, matches, good_matches, 2, keypoint1, keypoint2);

    vector<Point2f> goodPoints;
    vector<Point2f> matchedPoints;

    ////Get a better H
    //goodPoints放入左边图的goodmatch的点坐标，matchedPoints放入右边的
    for( int i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        goodPoints.push_back( keypoint1[good_matches[i].queryIdx ].pt );
        matchedPoints.push_back( keypoint2[good_matches[i].trainIdx ].pt );
    }

    ////3cb
    //再用这些点计算出H 4 pairs
    hom = findHomography( goodPoints, matchedPoints, 0 );
    //计算反向的H
    homlnv = hom.inv();

    ////3cc
    Mat img_matches;
    drawMatches( image1Display, keypoint1, image2Display, keypoint2,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    ////How many matching points?
    cout<<good_matches.size()<<endl;

    imshow("Match", img_matches);
//    imwrite("3.png", img_matches);
    //waitKey(0);
}

void Matcher::findInlier(Mat &H, vector<DMatch> &matches, vector<DMatch> &matches2, float inlierThrehold,
                            vector<KeyPoint> &keypoint1, vector<KeyPoint> &keypoint2) {

    float x, y;
    for(int i = 0; i < matches.size(); i++) {
        project(keypoint1[matches[i].queryIdx].pt.x, keypoint1[matches[i].queryIdx].pt.y, H, x, y);

        float rx = (keypoint2[matches[i].trainIdx].pt.x - x) * (keypoint2[matches[i].trainIdx].pt.x - x);
        float ry = (keypoint2[matches[i].trainIdx].pt.y - y) * (keypoint2[matches[i].trainIdx].pt.y - y);
        float distance = sqrt(rx + ry);
        if(distance < inlierThrehold) {
            matches2.push_back(matches[i]);
        }
    }
}

////4a
void Matcher::stitch(Mat &image1, Mat &image2, Mat &hom, Mat &homInv, Mat &stitchedImage, Mat& stitch) {

    //创建一个和右边的图一样大的4个角落的点的坐标
    vector<Point2f> image2_corners(4);
    image2_corners[0] = cvPoint(0,0); image2_corners[1] = cvPoint( image2.cols, 0 );
    image2_corners[2] = cvPoint( image2.cols, image2.rows ); image2_corners[3] = cvPoint( 0, image2.rows );
    //cout<< image2_corners<<endl;

    ////4aa
    vector<Point2f> projectedImage1_corners(4);
    //再把这四个角反向投影到左边，得到新的四个角
    for(int i = 0; i < 4; i++) {
        project(image2_corners[i].x, image2_corners[i].y, homInv, projectedImage1_corners[i].x, projectedImage1_corners[i].y);
    }
    //cout << projectedImage1_corners << endl;

    float minX = 0;
    float maxX = image1.cols;
    for(int i = 0; i < 4; i++) {
        if(projectedImage1_corners[i].x < minX) {
            minX = projectedImage1_corners[i].x;
        }
        if(projectedImage1_corners[i].x > maxX) {
            maxX = projectedImage1_corners[i].x;
        }
    }
    if(minX < 0) {
        minX = -minX;
    }

    float minY = 0;
    float maxY = image1.rows;
    for(int i = 0; i < 4; i++) {
        if(projectedImage1_corners[i].y < minY) {
            minY = projectedImage1_corners[i].y;
        }
        if(projectedImage1_corners[i].y > maxY) {
            maxY = projectedImage1_corners[i].y;
        }
    }
    if(minY < 0) {
        minY = -minY;
    }

    //Size
    int stitchRow, stitchCol;
    stitchRow = int(maxY) + int(minY);
    stitchCol = int(maxX) + int(minX);

    stitchedImage = Mat::zeros(stitchRow, stitchCol, image1.type());

    ////4ab
    //Put image1 into stitchedImage
    Mat imageROI;
    imageROI = stitchedImage(Rect((int)minX, (int)minY,image1.cols,image1.rows));
    image1.copyTo(imageROI);

    ////4ac
    for(int i = 0; i < stitchedImage.rows; i++) {
        for(int j = 0; j <stitchedImage.cols; j++) {
            float x, y;
            project(j - minX, i - minY, hom, x, y);
            if(x >= 0 && x <= image2.cols && y >= 0 && y <= image2.rows) {
                if(stitchedImage.at<Vec3b>(i, j)[0] == 0 ||stitchedImage.at<Vec3b>(i, j)[1] == 0||stitchedImage.at<Vec3b>(i, j)[2] == 0  ){
                    stitchedImage.at<Vec3b>(i, j) = image2.at<Vec3b>((int)y, (int)x);
                } else{
                    ////overlap
                    stitchedImage.at<Vec3b>(i, j) = 0.5*(stitchedImage.at<Vec3b>(i, j)) + 0.5*(image2.at<Vec3b>((int)y, (int)x));
                }

            }
        }
    }
//    for(int i = 0; i < stitchedImage.rows; i++) {
//        for(int j = 0; j <stitchedImage.cols; j++) {
//            float x, y;
//            //先平移在投影，将右图投影到左图上
//            project(j - minX, i - minY, hom, x, y);//找到右图点在左图的xy，在替换掉
//            if(x >= 0 && x <= image2.cols && y >= 0 && y <= image2.rows) {
//                if(image2.at<Vec3b>((int)y, (int)x)[0] == 0 ||image2.at<Vec3b>((int)y, (int)x)[1] == 0||image2.at<Vec3b>((int)y, (int)x)[2] == 0  ) continue;
//                stitchedImage.at<Vec3b>(i, j) = image2.at<Vec3b>((int)y, (int)x);
//            }
//        }
//    }
    stitch = stitchedImage;
}