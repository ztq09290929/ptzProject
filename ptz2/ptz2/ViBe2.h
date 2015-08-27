#pragma once
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#define NUM_SAMPLES 20		//每个像素点的样本个数
#define MIN_MATCHES 2		//#min指数，有两个以上灰度值相近，则认为是背景
#define RADIUS 25		//Sqthere半径，判断两个灰度值是否相近
#define SUBSAMPLE_FACTOR 8//子采样概率
#define VOTES 1				//邻域像素投票阈值，认为是背景的
#define RADIUS_NBHD	2	//搜索邻域圆的半径


class ViBe_BGS
{
public:
	ViBe_BGS(void);
	~ViBe_BGS(void);

	void init(const Mat& _image, Mat frame);   //初始化
	void processFirstFrame(const Mat& _image); //利用第一帧图像建立背景
	void testAndUpdate(const std::vector<cv::Point3f>& _image);  //利用当前帧更新前景

	Mat getMask(void){ return m_mask; };
	Mat getFore(void){ return m_fore; };


private:
	unsigned char ***samples;//此三维数组既保存20个背景，也保存前景计数器

	unsigned short m_NbhdPoints[(RADIUS_NBHD * 2 - 1)*(RADIUS_NBHD * 2 - 1)+4+2][2];//存储圆形邻域内的点
	uchar getNbhdPoints(float row, float col);//获取以RADIUS_NBHD为半径的园内的所有像素点坐标

	int imgRows;//存储全景图的尺寸大小，即为背景样本的尺寸，方便释放数组samples
	int imgCols;

	Mat m_pano;//存储灰度全景图
	Mat m_mask;//全景图中显示前景
	Mat m_fore;//前景
};