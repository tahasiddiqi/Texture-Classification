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

class LBPFeatures
{
public:
	Mat LBP(Mat src_image);
	Mat histogram(const cv::Mat& src);
};

