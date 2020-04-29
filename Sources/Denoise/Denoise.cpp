// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#define OPENCV_ENABLE_NONFREE
#include "opencv2/xphoto.hpp"
#include "opencv2/photo.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/videoio/registry.hpp>

#include <string>
#include <vector>

#include <iostream>
#include <experimental/filesystem>

#define OPENCV_VIDEOIO_PRIORITY_INTEL_MFX = 0
#define OPENCV_VIDEOIO_PRIORITY_LIST = FFMPEG

namespace filesys = std::experimental::filesystem;
using namespace cv;
using namespace std;


int main(int argc, char **argv)
{
/*
	cout << "List of Backends" << endl;
	std::vector<VideoCaptureAPIs> list = cv::videoio_registry::getBackends();
	for (auto l : list)
		cout << cv::videoio_registry::getBackendName(l) << endl;

	cout << "List of Writer Backends" << endl;
	list.clear();
	list = cv::videoio_registry::getWriterBackends();
	for (auto l : list)
		cout << cv::videoio_registry::getBackendName(l) << endl;

	cout << "List of Reader Backends" << endl;
	list.clear();
	list = cv::videoio_registry::getStreamBackends();
	for (auto l : list)
		cout << cv::videoio_registry::getBackendName(l) << endl;

	//return -1;
*/

	const std::string about =
		"Denoiser\n"
		"Exemple :\n"
		"Denoise.exe --input=c:/images/input --output=c:/images/sortie\n"
		"Fichiers acceptés : \n"
		"	Images jpg, png, tif\n"
		"	Videos mov, mp4, avi";

	const std::string keys =
		"{ h help    |             | print this help message }"
		"{ i input   | Input       | dossier entrée }"
		"{ o output  | Output      | dossier sortie }";
	CommandLineParser parser(argc, argv, keys);
	parser.about(about);
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	//std::string mode = parser.get<std::string>("color");
	std::string input = parser.get<std::string>("input");
	std::string output = parser.get<std::string>("output");


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
		string outnameimg = output + "/" + "denoise_" + name + ext;
		string outnamevideo = output + "/" + "denoise_" + name + ".mov";//ext;

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
			img = imread(file); // , cv::IMREAD_GRAYSCALE);
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
				video.open(outnamevideo,VideoWriter::fourcc('A', 'V', 'C', '1'), fps, Size(frameWidth, frameHeight)); //
				cap.read(img);
			}
		}
		cv::Mat res(img.size(), img.type());
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


			cv::cvtColor(img, img, COLOR_BGR2GRAY);

			cv::xphoto::bm3dDenoising(img, color, 15.0, 4, 16);

			/*
			if (mode=="grey")
				fastNlMeansDenoising(img, color, 10, 7, 21);
			else
				fastNlMeansDenoisingColored(img, color, 10, 7, 21);
			*/

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

