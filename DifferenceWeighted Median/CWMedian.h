#pragma once
#include <opencv2/opencv.hpp>
#include <algorithm>
#include "Utilities.h"

uchar CalculateCenterWeights(const cv::Mat& img, int pv, int filter_size)
{
	int totalPixels = img.rows * img.cols;
	uchar* dataArray = new uchar[totalPixels];

	int index = 0;
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			dataArray[index] = img.at<uchar>(i, j);
			index++;
		}
	}

	std::sort(dataArray, dataArray + totalPixels);

	std::vector<int> median_arr;
	
	int img_i = (filter_size - 1) / 2;

	for(int i = 0; i < filter_size - 1; i++)
	{
		median_arr.push_back(img.at<uchar>(img_i, img_i));
	}

	for (int i = 0; i < totalPixels; ++i) {
		median_arr.push_back(dataArray[i]);
	}

	std::sort(median_arr.begin(), median_arr.end());

	int mid = (median_arr.size() + 1) / 2;
	int median = median_arr[mid - 1];

	delete[] dataArray;

	return median;
}

cv::Mat Center_Weighted_Median(cv::Mat& src, int filter_size)
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
			buffer.at<uchar>(i - pad_size, j - pad_size) = CalculateCenterWeights(img_box, img.at<uchar>(i, j), filter_size);
		}
	}

	return buffer;
}