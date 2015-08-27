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

class ObjectAndKF//���������
{
public:
	vector<cv::Point> m_vecCenters;

	void Init();//��ʼ���������˲�������Ҫ����m_vecCenters�еĵ�һ����
	cv::Point Predict();//ʹ�ÿ������˲���Ԥ����һ���λ�ã�������һ��
	void Correct(int num);//ʹ��ͼ������ʵ�����ĵ������˲���

	inline cv::Point GetLastPoint()//��ȡm_vecCenters�е����һ����
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

class CBlob//���������
{
public:
	void BlobDetecter(const cv::Mat& _binImg, cv::Mat& _outputImg);//Ѱ��ǰ�������еĽϴ�������������Ӿ��κ����ĵ㣬�������ͼ���ϻ��ƾ���
	void ClassifyCenters(Mat& _outputImg);//��ǰ֡�ľ������Ĺ��ൽ��������
	void DrawPaths(cv::Mat& _outputImg);//�����켣
private:
	void FindNearstPointKal(cv::Point pLast, int& id, vector<int>& index);//���ÿ������˲�������Ԥ����һ��λ�ã����ڸ���Ѱ����һ��
	void FindNearstPoint(cv::Point pLast, int& id, vector<int>& index);//����һ�㷽����ȡĳһ��ĸ����ĵ�
	list<ObjectAndKF> m_listCenters;//��һ������洢��ͬ�����壬����ÿһ��ObjectAndKF�������һ�����壬ÿһ�����嶼��Ҫ�и��������˲���
	vector<cv::Point> m_centers;//ÿһ֡�ҵ��ľ������ĵ㶼��������
};
