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
#define RADIUS_NBHD	2	//��������뾶

#define USE_MNEW 1		//�µ�ViBe���ÿ�ƥ�䣬�ɵ�ViBe���õ�ƥ��
#define USE_M1 1		//����3*3����
#define USE_M2 0		//����Բ������

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

	std::vector<cv::Point2i> getNbhdPoints(float row,float col);//��ȡ��RADIUS_NBHDΪ�뾶��԰�ڵ��������ص�����

private:
	unsigned char ***samples;//����ά����ȱ���20��������Ҳ����ǰ��������

	int imgRows;//�洢ȫ��ͼ�ĳߴ��С����Ϊ���������ĳߴ磬�����ͷ�����samples
	int imgCols;

	Mat m_mask;//ȫ��ͼ����ʾǰ��
	Mat m_fore;//ǰ��
};