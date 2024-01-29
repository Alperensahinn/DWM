#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace cv;

float Gaussian(float x, float mu, float sigma) {
    float coefficient = 1.0f / (sigma * std::sqrt(2.0f * M_PI));
    float exponent = -0.5f * std::pow((x - mu) / sigma, 2); //can we opt this by doing (x*x) instead of pow
    float result = coefficient * std::exp(exponent);
    return result;
}

void AddPadding(cv::Mat& src, int filter_size) {
    int pad_size = (filter_size - 1) / 2;
    cv::copyMakeBorder(src, src, pad_size, pad_size, pad_size, pad_size, cv::BORDER_REPLICATE);
}

cv::Mat AddSaltAndPepperNoise(const cv::Mat& image, float probability) {
    cv::Mat noisyImage = image.clone();
    int totalPixels = image.total();

    int numSalt = static_cast<int>(std::ceil(probability * totalPixels * 0.5));
    int numPepper = static_cast<int>(std::ceil(probability * totalPixels * 0.5));

    for (int i = 0; i < numSalt; ++i) {
        int x = rand() % image.rows;
        int y = rand() % image.cols;
        //noisyImage.at<uchar>(x, y) = 255; // Salt noise

        std::random_device rd;
        std::mt19937 gen(rd());

        std::uniform_int_distribution<> distribution(253, 255);

        int randomNumber = distribution(gen);

        noisyImage.at<uchar>(x, y) = randomNumber;
    }

    for (int i = 0; i < numPepper; ++i) {
        int x = rand() % image.rows;
        int y = rand() % image.cols;
        //noisyImage.at<uchar>(x, y) = 0; // Pepper noise

        std::random_device rd;
        std::mt19937 gen(rd());

        std::uniform_int_distribution<> distribution(0, 2);

        int randomNumber = distribution(gen);

        noisyImage.at<uchar>(x, y) = randomNumber;
    }

    return noisyImage;
}

Scalar getMSSIM(const Mat& i1, const Mat& i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d = CV_32F;
    Mat I1, I2;
    i1.convertTo(I1, d);            // cannot calculate on one byte large values
    i2.convertTo(I2, d);
    Mat I2_2 = I2.mul(I2);        // I2^2
    Mat I1_2 = I1.mul(I1);        // I1^2
    Mat I1_I2 = I1.mul(I2);        // I1 * I2
    /*************************** END INITS **********************************/
    Mat mu1, mu2;                   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);
    Mat mu1_2 = mu1.mul(mu1);
    Mat mu2_2 = mu2.mul(mu2);
    Mat mu1_mu2 = mu1.mul(mu2);
    Mat sigma1_2, sigma2_2, sigma12;
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);                 // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);                 // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    Mat ssim_map;
    divide(t3, t1, ssim_map);        // ssim_map =  t3./t1;
    Scalar mssim = mean(ssim_map);   // mssim = average of ssim map
    return mssim;
}