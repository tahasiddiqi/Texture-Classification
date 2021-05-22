#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <string>
#include <sstream>

using namespace cv;
using namespace cv::ml;
using namespace std;


class SVMModel
{
public:
public:
	cv::String path;

	void trainLBP(String path);

	int predict(const cv::Mat& inp_img);

	float getAccuracy(String dir_path);
private:

	Mat histogram(const cv::Mat& src, cv::Mat& dst, Mat& b_hist);

	Mat LBP(Mat src_image, cv::Mat& lbp);

	Mat LBP_hist_features(const cv::Mat& src, cv::Mat& hist_flat);

	void Create_database(const cv::String path, cv::Mat& data, cv::Mat& label);

	int getLabel(char ch);
};

