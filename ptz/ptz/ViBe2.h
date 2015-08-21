#pragma once
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#define NUM_SAMPLES 20		//ÿ�����ص����������
#define MIN_MATCHES 2		//#minָ��
#define RADIUS 20		//Sqthere�뾶
#define SUBSAMPLE_FACTOR 8//�Ӳ�������
#define VOTES 1				//��������ͶƱ��ֵ����Ϊ�Ǳ�����

class ViBe_BGS
{
public:
	ViBe_BGS(void);
	~ViBe_BGS(void);

	void init(const Mat _image, Mat frame);   //��ʼ��
	void processFirstFrame(const Mat _image);
	void testAndUpdate(std::vector<cv::Point3f> _image);  //����
	Mat getMask(void){ return m_mask; };
	Mat getFore(void){ return m_fore; };
	//void deleteSamples(){ delete samples; };

private:
	unsigned char ***samples;//����ά����ȱ���20��������Ҳ����ǰ��������
	//	float samples[1024][1024][NUM_SAMPLES+1];//����ÿ�����ص������ֵ

	/*
	Mat m_samples[NUM_SAMPLES];
	Mat m_foregroundMatchCount;*/

	int imgRows;//�洢ȫ��ͼ�ĳߴ��С����Ϊ���������ĳߴ磬�����ͷ�����
	int imgCols;

	Mat m_mask;//ȫ��ͼ����ʾǰ��
	Mat m_fore;//ǰ��
};