#pragma once
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#define NUM_SAMPLES 20		//ÿ�����ص����������
#define MIN_MATCHES 2		//#minָ��
#define RADIUS 25		//Sqthere�뾶
#define SUBSAMPLE_FACTOR 8//�Ӳ�������
#define VOTES 1				//��������ͶƱ��ֵ����Ϊ�Ǳ�����
#define RADIUS_NBHD	2	//��������뾶


class ViBe_BGS
{
public:
	ViBe_BGS(void);
	~ViBe_BGS(void);

	void init(const Mat& _image, Mat frame);   //��ʼ��
	void processFirstFrame(const Mat& _image);
	void testAndUpdate(const std::vector<cv::Point3f>& _image);  //����

	Mat getMask(void){ return m_mask; };
	Mat getFore(void){ return m_fore; };


private:
	unsigned char ***samples;//����ά����ȱ���20��������Ҳ����ǰ��������

	unsigned short m_NbhdPoints[(RADIUS_NBHD * 2 - 1)*(RADIUS_NBHD * 2 - 1)+4+2][2];//�洢Բ�������ڵĵ�
	uchar getNbhdPoints(float row, float col);//��ȡ��RADIUS_NBHDΪ�뾶��԰�ڵ��������ص�����

	int imgRows;//�洢ȫ��ͼ�ĳߴ��С����Ϊ���������ĳߴ磬�����ͷ�����samples
	int imgCols;

	Mat m_pano;//�洢�Ҷ�ȫ��ͼ
	Mat m_mask;//ȫ��ͼ����ʾǰ��
	Mat m_fore;//ǰ��
};