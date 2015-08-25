#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/stitching/stitcher.hpp"
#include <time.h>
#include <iostream>
using namespace cv;
using namespace std;
bool po = false;
int main()
{
	int count = 0;

	cv::VideoCapture cap;
	cap.open("F:\\˶ʿ�γ�ѧϰ\\��Ƶ��������\\ptz��Դ\\��Ƶ����\\output.avi");
	if (!cap.isOpened())
	{
		std::cout << "��Ƶ��ʧ�ܣ�" << std::endl;
		return -1;
	}
	cv::Mat frame;///�洢��ǰ֡�������ȡ��һ֡
	std::vector<cv::Mat> backImages;///�洢������4����ͼ
	cv::Mat pano;///�洢ȫ������
	while (1)
	{
		cap.read(frame);

		cout << count << endl;
		if (count>270&&(count==276||count%100==30||count==600)&&count<602)
		{
			backImages.push_back(frame.clone());
		}

	}
	//backImages.push_back(imread("images/test-276.jpg"));
	//backImages.push_back(imread("images/test-300.jpg"));
	//backImages.push_back(imread("images/test-400.jpg"));
	//backImages.push_back(imread("images/test-500.jpg"));
	//backImages.push_back(imread("images/test-600.jpg"));
	cout << backImages.size() << endl;
	cout << "Please wait..." << endl;
	Stitcher stitcher = Stitcher::createDefault(false);
	Stitcher::Status status = stitcher.stitch(backImages, pano);

	imwrite("images/pano1.jpg", pano);

	return 0;
}