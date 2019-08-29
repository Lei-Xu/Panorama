//
// Created by Lei Xu on 2019-01-27.
//

#include "descriptor.h"

Mat Descriptor::getPatch(Mat& src, KeyPoint keyPoint) {
    Mat temp;
    Mat gray = src;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    int left = (int)keyPoint.pt.y - 8;
    int top = (int)keyPoint.pt.x - 8;

    if(top < 0) top += 8;
    if(left < 0) left += 8;
    if((top + 16) > src.cols) top -= 8;
    if((left + 16) > src.rows) left -= 8;
    temp = gray(cv::Rect(top,left,16,16));
    return temp;
}

Mat Descriptor::getDescriptor(Mat &src, KeyPoint keyPoint) {
    Mat patch_gray;
    patch_gray = getPatch(src, keyPoint);
    int n = 3;
    Mat Kernel(n, n, CV_64F);
    guassian(n, n, 1, Kernel);
    //将patch套上高斯滤镜
    filter2D(patch_gray, patch_gray, patch_gray.depth(), Kernel);

    //做1个128 dimensional descriptor
    Mat orientation = Mat::zeros(1, 128, CV_32FC1);
    Mat gau_x = gaussian_x(3, 1.0);
    Mat gau_y = gaussian_y(3, 1.0);

    Mat Dx, Dy;
    filter2D(patch_gray, Dx, CV_32F, gau_x);
    filter2D(patch_gray, Dy, CV_32F, gau_y);
    Mat mag, angle;
    //得到magnitude angle, according Dx, Dy
    cartToPolar(Dx, Dy, mag, angle, 1);

    int theta_value;//做旋转不变，算出最大的方向向量，然后转到垂直向上（或一固定角度）需要转多少度
    theta_value = theta(mag, angle);


    for(int i = 0 ;i < 4; i++){
        int beginPoint_row =  4 * i;
        for(int j = 0 ;j < 4;j++){
            int beginPoint_col =  4 * j;
            // devid into 4*4 cell

            for(int row = 0 ;row < 4; row++){
                for(int col = 0; col < 4; col++){

                    int dir = (int)angle.at<float>(row+beginPoint_row, col + beginPoint_col)//find the angle in each cell
                              - theta_value;//为了旋转无关，每个角度向量都要减去theta
                    if(dir < 0){
                        dir += 360;
                    }

                    int temp = dir / 45;//把360->8,一个4*4中的8个直方图
                    orientation.at<float>(0, i * 8 * 4 + j * 8 + temp)
                            += mag.at<float>(row + beginPoint_row,col + beginPoint_col);//在key point处统计把每个向量分别相加
                }
            }
        }
    }

    for(int i = 0 ; i < 16 ; i++){
        float max = 0;
        for(int j = 0 ; j < 8; j++){
            max += orientation.at<float>(0, i * 8 + j) * orientation.at<float>(0, i * 8 + j);
        }
        double thr = sqrt(max) * 0.2;
        max = 0;

        for(int j = 0; j < 8; j++) {
            if(thr <= orientation.at<float>(0,  i * 8 + j)) {
                orientation.at<float>(0, i * 8 + j) = (float)thr;
            }
            max += orientation.at<float>(0, i * 8 + j) * orientation.at<float>(0, i * 8 + j);
        }
        max = sqrtf(max);
        for(int j =0 ; j< 8 ;j++){
            orientation.at<float>(0, i * 8 + j) = orientation.at<float>(0, i * 8 + j) / max;
        }
        //contrast invariant
    }
    return orientation;
}

int Descriptor::theta(Mat &patch_mag, Mat &patch_dir) {
    Mat orientation = Mat::zeros(1, 36, CV_32FC1);
    int theta;
    for(int i = 0; i < patch_dir.rows; i++) {
        for(int j = 0; j < patch_dir.cols; j++) {
            int temp = int(patch_dir.at<float>(i, j) / 10);
            orientation.at<float>(0, temp) += patch_mag.at<float>(i, j);
        }
    }
    int min = 0;
    float temp = -1;
    for(int col = 0; col < 36; col++) {
        if (temp < orientation.at<float>(0, col)) {
            temp = orientation.at<float>(0, col);
            min = col;
        }
    }
    theta = 10 * min;
    return theta;
}

Mat Descriptor::descriptor(Mat& img,
                            CV_OUT CV_IN_OUT vector<KeyPoint>& keypoints) {
    Mat descriptors;
    descriptors = Mat::zeros((int)keypoints.size(), 128, CV_32FC1);//做一个N*128的矩阵存放descriptors
    for(int i = 0 ; i < keypoints.size();i++){
        cv::Mat temp = getDescriptor(img, keypoints[i]);//每个key point的descriptor，128个值
        for(int col = 0 ; col < 128; col++){
            descriptors.at<float>(i , col) = temp.at<float>(0, col);//放入矩阵
        }
    }
    return descriptors;
}

void Descriptor::guassian(int sizex, int sizey, double sigma, Mat &kernel) {
    double pi = M_PI;
    double mins = 0;
    if((sizex % 2 == 0)||(sizey % 2 == 0)){
        cout << "you set wrong number" << endl;
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


Mat Descriptor::gaussian_x(int ksize, float sigma)
{
    Mat kernel = Gaussian(ksize, sigma); // 首先得到原始高斯核
    Mat kernel_x(ksize, ksize, CV_32F, Scalar(0.0)); // 定义一阶高斯核
    for (int x = -ksize/2; x <= ksize/2; ++x) // 若为5*5大小，则x = (-2:1:2)
    {
        for (int i = 0; i < ksize; ++i)
        {
            kernel_x.at<float>(i, x + ksize/2) = -x/(sigma * sigma) * kernel.at<float>(i, x + ksize/2);
        }
    }
    return kernel_x;
}

Mat Descriptor::gaussian_y(int kSize, float sigma)
{
    Mat kernel = Gaussian(kSize, sigma); // 首先得到原始高斯核
    Mat kernel_y(kSize, kSize, CV_32F, Scalar(0.0)); // 定义一阶高斯核
    for (int y = -kSize/2; y <= kSize/2; ++y) // 若为5*5大小，则y = (-2:1:2)
    {
        for (int i = 0; i < kSize; ++i)
        {
            kernel_y.at<float>(y + kSize/2, i) = -y/(sigma * sigma) * kernel.at<float>(y + kSize/2, i);
        }
    }
    return kernel_y;
}


Mat Descriptor::Gaussian(int ksize, float sigma)
{
    if (ksize % 2 == 0)
    {
        cerr << "invalid kernel size." << endl;
        return Mat(1, 1, CV_32F, Scalar(-1));
    }
    Mat kernel(ksize, ksize, CV_32F, Scalar(0.0));
    Mat kernel_1d(ksize, 1, CV_32F, Scalar(0.0));
    for (int x = -ksize/2; x <= ksize/2; ++x)
    {
        kernel_1d.at<float>(x + ksize/2, 0) = exp(-(x * x)/(2 * sigma * sigma)) / (sigma * sqrt(2 * CV_PI));
    }
    kernel = kernel_1d * kernel_1d.t(); // 这里用两个一维相乘得到
    return kernel;
}