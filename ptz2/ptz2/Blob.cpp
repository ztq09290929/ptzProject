#include "blob.h"

void CBlob::BlobDetecter(const cv::Mat& _binImg, cv::Mat& _outputImg)
{
	cv::Mat src = _binImg.clone();
	cv::Mat srcROI = src(cv::Rect(src.cols / 20, src.rows / 20, src.cols * 18 / 20, src.rows * 18 / 20));//ÇÐµôÍ¼ÏñµÄ±ßÔµ5%
	cv::Mat outROI = _outputImg(cv::Rect(_outputImg.cols / 20, _outputImg.rows / 20, _outputImg.cols * 18 / 20, _outputImg.rows * 18 / 20));

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(srcROI, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	std::vector<std::vector<cv::Point>> goodContours;
	for (auto i = contours.begin(); i != contours.end(); ++i)
	{
		if (i->size() > 60)
		{
			goodContours.push_back(*i);
		}
	}

	std::vector<cv::Rect> rects;
	std::vector<cv::Point> centers;
	for (auto i = goodContours.begin(); i != goodContours.end(); ++i)
	{
		cv::Rect temRect = cv::boundingRect(*i);
		rects.push_back(temRect);
		cv::Point center;
		center.x = temRect.x + temRect.width / 2;
		center.y = temRect.y + temRect.height / 2;
		centers.push_back(center);
	}

	for (auto i = rects.begin(); i != rects.end(); ++i)
	{
		cv::rectangle(outROI, *i, cv::Scalar(0, 0, 255), 2, 8);
	}
	//for (auto i = centers.begin(); i != centers.end(); ++i)
	//{
	//	cv::circle(_outputImg, *i, 5, cv::Scalar(0, 255, 0), 2, 8);
	//}
}
