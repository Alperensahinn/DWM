#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <string>

#include "Utilities.h"
#include "DWMedianAdaptiveSigma.h"
#include "CWMedian.h"

int main()
{
	float noise_ratio = 0.001f;
	int kernel_size = 7;

	std::string path = "KODAK\\kodim01.png";

	cv::Mat src = cv::imread(path);
	cv::Mat gray_src;
	cv::cvtColor(src, gray_src, cv::COLOR_BGR2GRAY);

	cv::Mat noisy_img = AddSaltAndPepperNoise(gray_src, noise_ratio);

	cv::Mat dw_median = Difference_Weighted_Median(gray_src, kernel_size);

	cv::Mat cw_median = Center_Weighted_Median(gray_src, kernel_size);

	cv::Mat opencv_median;
	cv::medianBlur(noisy_img, opencv_median, kernel_size);

	//SSIM Calculation
	cv::Scalar ssim_scalar = getMSSIM(gray_src, dw_median);
	double ssim_dw_median = ssim_scalar.val[0];
	std::cout << "DW Median SSIM: " << ssim_dw_median << std::endl;

	ssim_scalar = getMSSIM(gray_src, cw_median);
	double ssim_cw_median = ssim_scalar.val[0];
	std::cout << "CW Median SSIM: " << ssim_cw_median << std::endl;

	ssim_scalar = getMSSIM(gray_src, opencv_median);
	double ssim_opencv_median = ssim_scalar.val[0];
	std::cout << "Median SSIM: " << ssim_opencv_median << std::endl;

	//PSNR Calculation
	cv::Mat dw_median_32f;
	gray_src.convertTo(gray_src, CV_32F);
	dw_median.convertTo(dw_median_32f, CV_32F);
	double psnr_dw_median = cv::PSNR(gray_src, dw_median_32f);
	std::cout << "DW Median PSNR: " << psnr_dw_median << std::endl;

	cv::Mat cw_median_32f;
	gray_src.convertTo(gray_src, CV_32F);
	cw_median.convertTo(cw_median_32f, CV_32F);
	double psnr_cw_median = cv::PSNR(gray_src, cw_median_32f);
	std::cout << "CW Median PSNR: " << psnr_cw_median << std::endl;

	cv::Mat oepncv_median_32f;
	gray_src.convertTo(gray_src, CV_32F);
	opencv_median.convertTo(oepncv_median_32f, CV_32F);
	double psnr_opencv_median = cv::PSNR(gray_src, oepncv_median_32f);
	std::cout << "PSNR: " << psnr_opencv_median << std::endl;

	cv::imshow("DW Median", dw_median);
	cv::imshow("CW Median", cw_median);
	cv::imshow("Median", opencv_median);

	cv::waitKey(0);

	src.release();
	gray_src.release();
	noisy_img.release();
	opencv_median.release();
	oepncv_median_32f.release();

	cv::destroyAllWindows();
}