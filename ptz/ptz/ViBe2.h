#pragma once
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#define NUM_SAMPLES 20		//每个像素点的样本个数
#define MIN_MATCHES 2		//#min指数
#define RADIUS 20		//Sqthere半径
#define SUBSAMPLE_FACTOR 16	//子采样概率


class ViBe_BGS
{
public:
	ViBe_BGS(void);
	~ViBe_BGS(void);

	void init(const Mat _image, Mat frame);   //初始化
	void processFirstFrame(const Mat _image);
	void testAndUpdate(std::vector<cv::Point3f> _image);  //更新
	Mat getMask(void){ return m_mask; };
	Mat getFore(void){ return m_fore; };
	//void deleteSamples(){ delete samples; };

private:
	unsigned char ***samples;//此三维数组既保存20个背景，也保存前景计数器
	//	float samples[1024][1024][NUM_SAMPLES+1];//保存每个像素点的样本值

	/*
	Mat m_samples[NUM_SAMPLES];
	Mat m_foregroundMatchCount;*/

	int imgRows;
	int imgCols;

	Mat m_mask;
	Mat m_fore;
};