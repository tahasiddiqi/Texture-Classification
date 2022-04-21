#include "Utils.h"
#include <fstream>

void Utils::showImage(Mat img)
{
	if (img.empty())
	{
		std::cout << "Could not read the image: " <<  std::endl;
		return;
	}
	imshow("Display window", img);
	waitKey(0); // Wait for a keystroke in the window
}

Mat Utils::rgb2Gray(Mat img)
{
	Mat gray;
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	return gray;
}


bool Utils::checkFileExist(const std::string& name)
{
	ifstream f(name.c_str());
	return f.good();
}
