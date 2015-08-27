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

class ObjectAndKF
{
public:
	void Init();
	vector<cv::Point> m_vecCenters;
	inline cv::Point GetLastPoint()
	{
		return m_vecCenters.at(m_vecCenters.size() - 1);
	}

	cv::Point Predict();
	void Correct(int num);

private:
	const int stateNum = 4;
	const int measureNum = 2;

	KalmanFilter KF;
	Mat state; //state(x,y,detaX,detaY)  
	Mat processNoise;
	Mat measurement;    //measurement(x,y)  
};

class CBlob
{
public:
	void BlobDetecter(const cv::Mat& _binImg, cv::Mat& _outputImg);
	void ClassifyCenters();
	void DrawPaths(cv::Mat& _outputImg);
private:
	void FindNearstPointKal(cv::Point pLast, int& id, vector<int>& index);
	void FindNearstPoint(cv::Point pLast, int& id, vector<int>& index);
	list<ObjectAndKF> m_listCenters;///注意，点的坐标都是取过ROI之后的坐标
	vector<cv::Point> m_centers;
};
