#include "SVMModel.h"
#include "Utils.h"


void SVMModel::trainLBP(String path)
{
	Mat data, label;
	cout << "Initiating data Generation" << endl;
	Create_database(path, data, label);
	/*cout << label << endl;*/
	cout << "Data Generation Completed" << endl;
	cout << "Initiating training." << endl;
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm->train(data, ROW_SAMPLE, label);
	svm->save("trained-svm.xml");
	cout << "Training completed." << endl;

}

int SVMModel::predict(const cv::Mat& inp_img)

{
	int prediction;
	Utils utils;
	Mat hist_flat;
	String model_name = "trained-svm.xml";

	if(!utils.checkFileExist(model_name))
	{
		cout << "Model does not exist" << endl;
		return -1;
	}

	//Loading Trained Model
	Ptr<SVM> svm = Algorithm::load<SVM>(model_name);
	vector<cv::String> fn;
	
	//Preprocessing
	hist_flat = LBP_hist_features(inp_img, hist_flat);
	prediction = svm->predict(hist_flat);
	/*cout << prediction << endl;*/

	return prediction;
}

float SVMModel::getAccuracy(String dir_path)
{
	vector<string> fn;
	Mat hist_flat;
	char ch;
	int label_id, prediction, accurate_img_cntr = 0, total_img_cntr = 0;

	cv::glob(dir_path, fn, true); // recurse
	for (size_t k = 0; k < fn.size(); ++k)
	{
		total_img_cntr = fn.size();

		//Get Label
		label_id = getLabel(fn[k][dir_path.length() + 1]);
		/*cout << label_id << endl;*/

		//Reading image
		cv::Mat im = cv::imread(fn[k]);
		if (im.empty()) continue; //only proceed if successful
		prediction = predict(im);
		/*cout << prediction << endl;*/

		//Incrementing accurate result
		if (prediction == label_id) {
			accurate_img_cntr += 1;
		}
	}
	return (accurate_img_cntr/total_img_cntr)*100;
}

Mat SVMModel::histogram(const cv::Mat& src, cv::Mat& dst, Mat& b_hist)

{
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
	return b_hist;
}

Mat SVMModel::LBP(Mat src_image, cv::Mat& lbp)

{
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

Mat SVMModel::LBP_hist_features(const cv::Mat& src, cv::Mat& hist_flat)

{
	cv::Mat lbp(src.rows, src.cols, CV_8UC1);
	Mat hist, Gray, b_hist;

	lbp = LBP(src, lbp);
	b_hist = histogram(lbp, hist, b_hist);

	hist_flat = b_hist.reshape(1, 1);

	return hist_flat;
}

void SVMModel::Create_database(const cv::String path, cv::Mat& data, cv::Mat& label)
{
	vector<string> fn;
	Mat hist_flat;
	char ch;
	int label_id;

	cv::glob(path, fn, true); // recurse
	for (size_t k = 0; k < fn.size(); ++k)
	{
		label_id = getLabel(fn[k][path.length() + 1]);
		cv::Mat im = cv::imread(fn[k]);
		if (im.empty()) continue; //only proceed if successful
		hist_flat = LBP_hist_features(im, hist_flat);
		data.push_back(hist_flat);
		label.push_back(label_id);
	}
}

int SVMModel::getLabel(char ch)
{
	/*cout << fn[k] << endl;
	cout << ch << endl;*/
	int label_id = (int)ch - 48; //ASCII value to int
	/*cout << label_id << endl;*/
	return label_id;
}
