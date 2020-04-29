// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

//#include <opencv2/dnn.hpp>
#include "opencv2/dnn_superres.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <experimental/filesystem>

namespace filesys = std::experimental::filesystem;
using namespace cv;
using namespace cv::dnn;
using namespace std;


int main(int argc, char **argv)
{
	const std::string about =
		"Augmentation de la résolution d'images\n"
		"Arguments : dossiers entrée et sortie, algorythme, echelle\n"
		"Exemple :\n"
		"SuperRes.exe --input=c:/images/input --output=c:/images/sortie --algo=fsrcnn --scale=4\n"
		"Fichiers acceptés : \n"
		"	Images jpg, png, tif\n"
		"	Videos mov, mp4, avi";

	const std::string keys =
		"{ h help   |             | print this help message }"
		"{ i input  | Input       | dossier entrée }"
		"{ o output | Output      | dossier sortie }"
		"{ a algo   | espcn       | edsr, espcn, fsrcnn or lapsrn }"
		"{ s scale  | 2           | 2, 3, 4 or 8 }";
	CommandLineParser parser(argc, argv, keys);
	parser.about(about);
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	std::string algo = parser.get<std::string>("algo");
	std::string input = parser.get<std::string>("input");
	std::string output = parser.get<std::string>("output");
	int scale = parser.get<int>("scale");

	if ((algo != "edsr") && (algo != "espcn") && (algo != "fsrcnn") && (algo != "lapsrn")) {
		cout << "Unknown algo" << endl;
		return 1;
	}

	if ((scale!=2) && (scale != 3) && (scale != 4) && (scale != 8)) {
		cout << "Incorrect scale" << endl;
		return 2;
	}

	ifstream ifile;
	string model = algo;

	std::for_each(model.begin(), model.end(), [](char & c) {
		c = ::toupper(c);
	});
	model = "Models/" + model + "_x" + to_string(scale) + ".pb";
	ifile.open(model);
	if (!ifile) {
		algo = "fsrcnn";
		model = "Models/FSRCNN_x2.pb";
		scale = 2;
	}

	cv::dnn_superres::DnnSuperResImpl sr;
	Net net = dnn::readNetFromTensorflow(model);
	net.setPreferableBackend(DNN_BACKEND_CUDA);
	net.setPreferableTarget(DNN_TARGET_CUDA);

	


	if (!parser.check())
	{
		parser.printErrors();
		return 1;
	}

	std::string folder(input + "/*.*");
	std::vector<cv::String> filenames;
	glob(folder, filenames, false);

	string type = "";
	Mat img, color;
	VideoCapture cap;
	VideoWriter video;

	for (auto file : filenames) {

		string ext = filesys::path(file).extension().string();
		string name = filesys::path(file).stem().string();
		string outnameimg = output + "/" + "superres_" + name + ext;
		string outnamevideo = output + "/" + "superres_" + name + ".mov";//ext;

		type = "nul";
		if ((ext == ".jpg") || (ext == ".png") || (ext == ".tif"))
			type = "img";


		if ((ext == ".mov") || (ext == ".mp4") || (ext == ".avi"))
			type = "vdo";


		if (type == "nul") {
			cout << "Unknown extension :" << ext << endl;
			break;
		}

		if (type == "img") {
			img = imread(file);
			if (img.empty())
			{
				cout << "Can't read image from file: " << file << endl;
				break;
			}
		}

		if (type == "vdo") {
			cap.open(file);
			if (!cap.isOpened())
			{
				cout << "Unable to open video file :" << file << endl;
				break;
			}
			else {
				int frameWidth = cap.get(CAP_PROP_FRAME_WIDTH);
				int frameHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
				double fps = cap.get(CAP_PROP_FPS);
				int fourcc = cap.get(CAP_PROP_FOURCC);
				video.open(outnamevideo, VideoWriter::fourcc('A', 'V', 'C', '1'), fps, Size(frameWidth*scale, frameHeight*scale)); //VideoWriter::fourcc('H', 'E', 'V', '1')
			}
		}

		cout << endl;

		if (type == "vdo")
			cout << "Processing " << file << " -> " << outnamevideo << endl;
		else
			cout << "Processing " << file << " -> " << outnameimg << endl;

		int pos = 0, oldpos = -1;
		for (;;) {
			if (type == "vdo") {
				cap.read(img);
				if (img.empty())
					break;

				pos = floor((cap.get(CAP_PROP_POS_FRAMES) / cap.get(CAP_PROP_FRAME_COUNT)) * 100);
				if ((pos % 5 == 0) && (pos != oldpos)) {
					cout << pos << "% ";
					oldpos = pos;
				}

			}
			

			if (algo == "espcn" || algo == "lapsrn" || algo == "fsrcnn")
			{
				//Preprocess the image: convert to YCrCb float image and normalize
				Mat preproc_img;
				Mat ycrcb;
				cv::cvtColor(img, ycrcb, COLOR_BGR2YCrCb);
				ycrcb.convertTo(preproc_img, CV_32F, 1.0 / 255.0);

				//Split the image: only the Y channel is used for inference
				Mat ycbcr_channels[3];
				split(preproc_img, ycbcr_channels);

				Mat Y = ycbcr_channels[0];

				//Create blob from image so it has size 1,1,Width,Height
				cv::Mat blob;
				dnn::blobFromImage(Y, blob, 1.0);

				//Get the HR output
				net.setInput(blob);

				Mat blob_output = net.forward();

				//Convert from blob
				std::vector <Mat> model_outs;
				dnn::imagesFromBlob(blob_output, model_outs);
				Mat out_img = model_outs[0];

				//Reconstruct: upscale the Cr and Cb space and merge the three layer
				Mat orig_channels[3];
				split(preproc_img, orig_channels);

				Mat Cr, Cb;
				cv::resize(orig_channels[1], Cr, cv::Size(), scale, scale);
				cv::resize(orig_channels[2], Cb, cv::Size(), scale, scale);

				std::vector <Mat> channels;
				channels.push_back(out_img);
				channels.push_back(Cr);
				channels.push_back(Cb);

				Mat merged_img;
				merge(channels, merged_img);

				Mat merged_8u_img;
				merged_img.convertTo(merged_8u_img, CV_8U, 255.0);

				cv::cvtColor(merged_8u_img, color, COLOR_YCrCb2BGR);
			}
			
			else if (model == "edsr")
			{
				//BGR mean of the Div2K dataset
				Scalar mean = Scalar(103.1545782, 111.561547, 114.35629928);

				//Convert to float
				Mat float_img;
				img.convertTo(float_img, CV_32F, 1.0);

				//Create blob from image so it has size [1,3,Width,Height] and subtract dataset mean
				cv::Mat blob;
				dnn::blobFromImage(float_img, blob, 1.0, Size(), mean);

				//Get the HR output
				net.setInput(blob);
				Mat blob_output = net.forward();

				//Convert from blob
				std::vector <Mat> model_outs;
				dnn::imagesFromBlob(blob_output, model_outs);

				//Post-process: add mean.
				Mat(model_outs[0] + mean).convertTo(color, CV_8U);
			}
			

			if (type == "img")
				break;

			if (type == "vdo")
				video.write(color);
		}
		if (type == "img")
			cv::imwrite(outnameimg, color);
		if (type == "vdo") {
			cap.release();
			video.release();
		}

		cout << endl;
	}


	return 0;
}

