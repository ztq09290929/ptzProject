#pragma once
#include "opencv2/video/background_segm.hpp"
#include "opencv2/legacy/blobtrack.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc_c.h>
#include <list>
#include <opencv/cv.h>  

using namespace std;
using namespace cv;

class ObjectAndKF//物体跟踪类
{
public:
	vector<cv::Point> m_vecCenters;

	void Init();//初始化卡尔曼滤波器，需要传入m_vecCenters中的第一个点
	cv::Point Predict();//使用卡尔曼滤波器预测下一点的位置，返回下一点
	void Correct(int num);//使用图像中真实的中心点修正滤波器

	inline cv::Point GetLastPoint()//获取m_vecCenters中的最后一个点
	{
		return m_vecCenters.at(m_vecCenters.size() - 1);
	}

private:
	const int stateNum = 4;
	const int measureNum = 2;

	KalmanFilter KF;
	Mat state; //state(x,y,detaX,detaY)  
	Mat processNoise;
	Mat measurement;    //measurement(x,y)  
};

class CBlob//物体查找类
{
public:
	void BlobDetecter(const cv::Mat& _binImg, cv::Mat& _outputImg);//寻找前景物体中的较大轮廓，求其外接矩形和中心点，并在输出图像上绘制矩形
	void ClassifyCenters(Mat& _outputImg);//当前帧的矩形中心归类到各个物体
	void DrawPaths(cv::Mat& _outputImg);//画出轨迹
private:
	void FindNearstPointKal(cv::Point pLast, int& id, vector<int>& index);//利用卡尔曼滤波方法，预估下一点位置，并在附近寻找下一点
	void FindNearstPoint(cv::Point pLast, int& id, vector<int>& index);//利用一般方法求取某一点的附近的点
	list<ObjectAndKF> m_listCenters;//用一个链表存储不同的物体，其中每一个ObjectAndKF对象就是一个物体，每一个物体都需要有个卡尔曼滤波器
	vector<cv::Point> m_centers;//每一帧找到的矩形中心点都存在这里
};
