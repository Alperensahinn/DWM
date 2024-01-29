#pragma once
#include <opencv2/opencv.hpp>
#include <algorithm>
#include "Utilities.h"

uchar CalculateWeights(const cv::Mat& img, int pv)
{
	int totalPixels = img.rows * img.cols;
	int* dataArray = new int[img.rows * img.cols];

	int index = 0;
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			dataArray[index] = img.at<uchar>(i, j);
			index++;
		}
	}

	//sigma cal
	float m = 0;
	for (int i = 0; i < totalPixels; ++i) 
	{
		m = m + (float) dataArray[i];
	}

	m = m / totalPixels;

	float s_sqr = 0;

	for (int i = 0; i < totalPixels; ++i)
	{
		s_sqr += (dataArray[i] - m) * (dataArray[i] - m);
	}

	s_sqr = s_sqr / (totalPixels - 1);

	float s = sqrt(s_sqr);

	if (s < 30)
		s = 30;

	//diff cal
	std::sort(dataArray, dataArray + totalPixels);

	int* diff_arr = new int[totalPixels];


	for (int i = 0; i < totalPixels; ++i) {
		diff_arr[i] = static_cast<int>(std::floor(Gaussian(std::abs(pv - dataArray[i]), 0, s * 1.5f) * 10000));
	}

	std::vector<int> median_arr;

	for (int i = 0; i < totalPixels; ++i) {
		if (diff_arr[i] > 10)
		{
			for (int j = 0; j < diff_arr[i]; ++j) {
				median_arr.push_back(dataArray[i]);
			}
		}
		else
		{
			for (int j = 0; j < 10; ++j) {
				median_arr.push_back(dataArray[i]);
			}
		}
	}

	int middleIndex = median_arr.size() / 2;

	delete[] dataArray;
	delete[] diff_arr;

	if (median_arr.size() > 0)
		return median_arr[middleIndex];
	else
		return pv;
}

cv::Mat Difference_Weighted_Median(cv::Mat& src, int filter_size)
{
	cv::Mat img = src.clone();
	cv::Mat buffer = src.clone();
	AddPadding(img, filter_size);

	int pad_size = (filter_size - 1) / 2;

	for (int i = pad_size; i < img.rows - pad_size; i++)
	{
		for (int j = pad_size; j < img.cols - pad_size; j++)
		{
			cv::Mat img_box = img(cv::Rect(j - pad_size, i - pad_size, filter_size, filter_size));
			buffer.at<uchar>(i - pad_size, j - pad_size) = CalculateWeights(img_box, img.at<uchar>(i, j));
		}
	}

	return buffer;
}