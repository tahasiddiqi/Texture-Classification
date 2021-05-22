// TextureDetection.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <iostream>
#include "Utils.h"
#include "LBPFeatures.h"
#include "SVMModel.h"

using namespace cv;
using namespace std;

String training_dir_path = "./database2/train";
String testing_dir_path = "./database2/test";
String validation_dir_path = "./database2/validation";
String test_image_path = "./database2/test/4 (6).png";


//MILESTONE 1
//LBP Feature Extraction using OpenCV and C++
void lbpFeaturedetection() {
	Utils utils;
	Mat img = cv::imread(test_image_path);
	LBPFeatures lbpFeatures;

	Mat lbp = lbpFeatures.LBP(img);
	Mat hist = lbpFeatures.histogram(lbp);

	cv::imwrite("lbp.png", lbp);
	cv::imwrite("hist.png", hist);
	cv::imwrite("input_img.png", img);
}


//MILESTONE 2
//OPENCV SVM CLASSIFIER
void trainAndTest() {
	Mat img;
	Utils utils;
	SVMModel model;

	//Train model
	model.trainLBP(training_dir_path);

	cout << endl;
	cout << "Calulating Model Accuracy " << endl;
	cout << "Model Accuracy is " << model.getAccuracy(training_dir_path) << "%" << endl;
	cout << endl;

	//Display image
	img = cv::imread(test_image_path);
	utils.showImage(img);

	//Test Model
	int prediction = model.predict(img);
	cout << "Pattern is of type " << prediction << endl;
}

int main()
{

	// MILESTONE 1
	//LBP Feature Extraction using OpenCV and C++
	lbpFeaturedetection();

	//MILESTONE 2 & 3
	//OPENCV SVM CLASSIFIER
	trainAndTest();
	
}




