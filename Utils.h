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

class Utils
{
public:
	static void showImage(Mat img);
	static Mat rgb2Gray(Mat img);
	static Mat rgba2Gray(Mat img);
	static bool checkFileExist(const std::string& name);
};

