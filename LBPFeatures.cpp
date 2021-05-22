#include "LBPFeatures.h"

Mat LBPFeatures::LBP(Mat src_image)
{
	cv::Mat lbp(src_image.rows, src_image.cols, CV_8UC1);;
	bool affiche = true;
	cv::Mat Image(src_image.rows, src_image.cols, CV_8UC1);

	if (src_image.channels() == 3)
		cvtColor(src_image, Image, cv::COLOR_BGR2GRAY);

	unsigned center = 0;
	unsigned center_lbp = 0;

	for (int row = 1; row < Image.rows - 1; row++)

	{

		for (int col = 1; col < Image.cols - 1; col++)

		{

			center = Image.at<uchar>(row, col);
			center_lbp = 0;

			if (center <= Image.at<uchar>(row - 1, col - 1))
				center_lbp += 1;

			if (center <= Image.at<uchar>(row - 1, col))
				center_lbp += 2;

			if (center <= Image.at<uchar>(row - 1, col + 1))
				center_lbp += 4;

			if (center <= Image.at<uchar>(row, col - 1))
				center_lbp += 8;

			if (center <= Image.at<uchar>(row, col + 1))
				center_lbp += 16;

			if (center <= Image.at<uchar>(row + 1, col - 1))
				center_lbp += 32;

			if (center <= Image.at<uchar>(row + 1, col))
				center_lbp += 64;

			if (center <= Image.at<uchar>(row + 1, col + 1))
				center_lbp += 128;
			lbp.at<uchar>(row, col) = center_lbp;
		}
	}
	return lbp;
}

Mat LBPFeatures::histogram(const cv::Mat& src)
{
	Mat hist, Gray, b_hist;
	//Calculate Histogram
	int histSize = 26; //256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	calcHist(&src, 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}
	imshow("Source image", src);
	imshow("calcHist Demo", histImage);
	waitKey();

	return histImage;
}
