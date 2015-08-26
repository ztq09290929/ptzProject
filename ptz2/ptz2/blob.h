#pragma once
#include "opencv2/video/background_segm.hpp"
#include "opencv2/legacy/blobtrack.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc_c.h>
#include <list>

using namespace std;
class CBlob
{
public:
	void BlobDetecter(const cv::Mat& _binImg, cv::Mat& _outputImg);
	void ClassifyCenters();
	void DrawPaths(cv::Mat& _outputImg);
private:
	void FindNearstPoint(cv::Point pLast,int& id,vector<int>& index);
	list<vector<cv::Point>> m_listCenters;///注意，点的坐标都是取过ROI之后的坐标
	vector<cv::Point> m_centers;
};



