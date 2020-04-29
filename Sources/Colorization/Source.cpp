// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include <string>
#include <vector>
#include <iostream>
#include <experimental/filesystem>

#define OPENCV_VIDEOIO_PRIORITY_INTEL_MFX = 0
#define OPENCV_VIDEOIO_PRIORITY_LIST = FFMPEG

namespace filesys = std::experimental::filesystem;
using namespace cv;
using namespace cv::dnn;
using namespace std;


// the 313 ab cluster centers from pts_in_hull.npy (already transposed)
static float hull_pts[] = {
	-90., -90., -90., -90., -90., -80., -80., -80., -80., -80., -80., -80., -80., -70., -70., -70., -70., -70., -70., -70., -70.,
	-70., -70., -60., -60., -60., -60., -60., -60., -60., -60., -60., -60., -60., -60., -50., -50., -50., -50., -50., -50., -50., -50.,
	-50., -50., -50., -50., -50., -50., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -30.,
	-30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -20., -20., -20., -20., -20., -20., -20.,
	-20., -20., -20., -20., -20., -20., -20., -20., -20., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10.,
	-10., -10., -10., -10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 10., 10., 10., 10., 10., 10., 10.,
	10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20.,
	20., 20., 20., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 40., 40., 40., 40.,
	40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50.,
	50., 50., 50., 50., 50., 50., 50., 50., 50., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60.,
	60., 60., 60., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 80., 80., 80.,
	80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 90., 90., 90., 90., 90., 90., 90., 90., 90., 90.,
	90., 90., 90., 90., 90., 90., 90., 90., 90., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 50., 60., 70., 80., 90.,
	20., 30., 40., 50., 60., 70., 80., 90., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -20., -10., 0., 10., 20., 30., 40., 50.,
	60., 70., 80., 90., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., -40., -30., -20., -10., 0., 10., 20.,
	30., 40., 50., 60., 70., 80., 90., 100., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., -50.,
	-40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., -60., -50., -40., -30., -20., -10., 0., 10., 20.,
	30., 40., 50., 60., 70., 80., 90., 100., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90.,
	100., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -80., -70., -60., -50.,
	-40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -90., -80., -70., -60., -50., -40., -30., -20., -10.,
	0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -100., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30.,
	40., 50., 60., 70., 80., 90., -100., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70.,
	80., -110., -100., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., -110., -100.,
	-90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., -110., -100., -90., -80., -70.,
	-60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., -110., -100., -90., -80., -70., -60., -50., -40., -30.,
	-20., -10., 0., 10., 20., 30., 40., 50., 60., 70., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0.
};

int main(int argc, char **argv)
{
	const std::string about =
		"Colorisation d'images\n"
		"Arguments : dossiers entrée et sortie\n"
		"Exemple :\n"
		"colorization.exe --input=c:/images/input --output=c:/images/sortie\n"
		"Fichiers acceptés : \n"
		"	Images jpg, png, tif\n"
		"	Videos mov, mp4, avi";

	const std::string keys =
		"{ h help   |         | print this help message }"
		"{ i input  | Input   | dossier entrée }"
		"{ o output | Output  | dossier sortie }";
	CommandLineParser parser(argc, argv, keys);
	parser.about(about);
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	
	std::string modelTxt = "Models/colorization_deploy_v2.prototxt"; //samples::findFile(parser.get<string>("proto"));
	std::string modelBin = "Models/colorization_release_v2.caffemodel"; // samples::findFile(parser.get<string>("model"));
	std::string input = parser.get<std::string>("input");
	std::string output = parser.get<std::string>("output");

	// fixed input size for the pretrained network
	const int W_in = 224;
	const int H_in = 224;
	Net net = dnn::readNetFromCaffe(modelTxt, modelBin);
	net.setPreferableBackend(DNN_BACKEND_CUDA);
	net.setPreferableTarget(DNN_TARGET_CUDA);

	// setup additional layers:
	int sz[] = { 2, 313, 1, 1 };
	const Mat pts_in_hull(4, sz, CV_32F, hull_pts);
	Ptr<dnn::Layer> class8_ab = net.getLayer("class8_ab");
	class8_ab->blobs.push_back(pts_in_hull);
	Ptr<dnn::Layer> conv8_313_rh = net.getLayer("conv8_313_rh");
	conv8_313_rh->blobs.push_back(Mat(1, 313, CV_32F, Scalar(2.606)));


	if (!parser.check())
	{
		parser.printErrors();
		return 1;
	}

	std::string folder(input+"/*.*");
	std::vector<cv::String> filenames;
	glob(folder, filenames, false);

	string type = "";
	Mat img, color;
	VideoCapture cap;
	VideoWriter video;

	for (auto file : filenames) {

		string ext = filesys::path(file).extension().string();
		string name = filesys::path(file).stem().string();
		string outnameimg = output + "/" + "color_" + name + ext;
		string outnamevideo = output + "/" + "color_" + name + ".mov";

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
				video.open(outnamevideo, VideoWriter::fourcc('A', 'V', 'C', '1'), fps , Size(frameWidth, frameHeight)); //VideoWriter::fourcc('M', 'P', '4', 'V')
			}
		}

		cout << endl;

		if (type=="vdo")
			cout << "Processing " << file << " -> " << outnamevideo <<endl;
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
			// extract L channel and subtract mean
			Mat lab, L, input;
			img.convertTo(img, CV_32F, 1.0 / 255);
			cv::cvtColor(img, lab, COLOR_BGR2Lab);
			cv::extractChannel(lab, L, 0);
			cv::resize(L, input, Size(W_in, H_in));
			input -= 50;

			// run the L channel through the network
			Mat inputBlob = blobFromImage(input);
			net.setInput(inputBlob);
			Mat result = net.forward();

			// retrieve the calculated a,b channels from the network output
			Size siz(result.size[2], result.size[3]);
			Mat a = Mat(siz, CV_32F, result.ptr(0, 0));
			Mat b = Mat(siz, CV_32F, result.ptr(0, 1));
			cv::resize(a, a, img.size());
			cv::resize(b, b, img.size());

			// merge, and convert back to BGR
			Mat chn[] = { L, a, b };
			cv::merge(chn, 3, lab);
			cv::cvtColor(lab, color, COLOR_Lab2BGR);
			color = color * 255;
			color.convertTo(color, CV_8U);

			//imshow("Img", color);

			if (type == "img")
				break;

			if (type == "vdo")
				video.write(color);
		}
		if (type=="img")
			cv::imwrite(outnameimg, color);
		if (type == "vdo") {
			cap.release();
			video.release();
		}

		cout << endl;
	}

	
	return 0;
}

