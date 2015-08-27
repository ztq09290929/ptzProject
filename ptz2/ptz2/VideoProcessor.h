#ifndef _VIDEOPROCESSOR_
#define  _VIDEOPROCESSOR_
#include <opencv2/opencv.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/stitching/stitcher.hpp"
#include <iostream>
#include <string>
#include <time.h>
#include "ViBe2.h"
#include "KeyPointMatch.h"
#include "blob2.h"


using namespace std;
using namespace cv;


class VideoProcessor
{
public:
inline	VideoProcessor():m_delay(1)//构造函数，初始化列表赋初值
	{
//		blob = cvCreateBlobTrackerAuto1();
	}
//inline	~VideoProcessor(){ delete blob; }

	void Init(std::string filename);//获取全景图像，获取全景图像的特征，用全景图像建立ViBe初始背景

	int SetInput(std::string filename);//设置输入视频文件名称，并打开视频
	void SetDelay(int d);//设置延迟

	void DisplayInput(std::string wn);//创建输入窗口
	void DisplayOutputFront(std::string wn);//创建输出前景窗口
	void DisplayOutputBack(std::string wn);//创建输出背景窗口

	void Run();//开始处理循环

private:
	cv::VideoCapture m_capture;//打开的视频

	std::string m_windowNameInput;//输入窗口名称
	std::string m_windowNameOutputFront;//输出前景窗口名称
	std::string m_windowNameOutputBack;//输出背景窗口名称

	cv::Mat m_frame;//存储当前读取的一帧图像
	cv::Mat m_foreImage;//前景图像
	cv::Mat m_backImage;//背景图像
	cv::Mat m_pano;//全景图像
	std::vector<cv::Point3f> m_curFrame;//将当前帧利用H矩阵，坐标变换到全景背景图之后的当前帧的所有点，其中Point3f的三个参数分别为x,y,gray

	ViBe_BGS vibe_bgs;//ViBe算法类
	KeyPointMatch keyPointMatch;//图像配准类
	CBlob cblob;//目标跟踪类

	int m_delay;//每两帧之间的时间间隔

};

#endif