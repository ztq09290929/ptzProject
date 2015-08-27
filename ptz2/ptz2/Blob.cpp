#include "blob.h"

void CBlob::BlobDetecter(const cv::Mat& _binImg, cv::Mat& _outputImg)
{
	cv::Mat src = _binImg.clone();
	cv::Mat srcROI = src(cv::Rect(src.cols / 20, src.rows / 20, src.cols * 18 / 20, src.rows * 18 / 20));//切掉图像的边缘5%
	cv::Mat outROI = _outputImg(cv::Rect(_outputImg.cols / 20, _outputImg.rows / 20, _outputImg.cols * 18 / 20, _outputImg.rows * 18 / 20));

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(srcROI, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	std::vector<std::vector<cv::Point>> goodContours;
	for (auto i = contours.begin(); i != contours.end(); ++i)
	{
		if (i->size() > 45)///人为设定的阈值，排除较小的轮廓
		{
			goodContours.push_back(*i);
		}
	}

	m_centers.clear();
	std::vector<cv::Rect> rects;
	for (auto i = goodContours.begin(); i != goodContours.end(); ++i)
	{
		cv::Rect temRect = cv::boundingRect(*i);
		rects.push_back(temRect);
		cv::Point center;
		center.x = temRect.x + temRect.width / 2;
		center.y = temRect.y + temRect.height / 2;
		m_centers.push_back(center);
	}


	for (auto i = rects.begin(); i != rects.end(); ++i)
	{
		cv::rectangle(outROI, *i, cv::Scalar(0, 255, 0), 2, 8);
	}

}

void CBlob::FindNearstPoint(cv::Point pLast, int& id, vector<int>& index)
{
	double dist = 0;
	double minDist = 10000;
	int count = 0;
//	if (m_centers.empty())cout << "m_centers为空" << endl;
	for (auto i = m_centers.begin(); i != m_centers.end(); ++i)
	{
		dist = sqrt((pLast.x - (i->x))*(pLast.x - (i->x)) + (pLast.y - (i->y))*(pLast.y - (i->y)));

		if (dist < minDist)
		{
			minDist = dist;
			id = count;
		}
		++count;
	}
	if (minDist < 100)
	{
		index[id] = 0;
	}
	else
	{
		id =  -1;
	}
}

void CBlob::ClassifyCenters()
{
	if (m_listCenters.empty())
	{
		for (auto i = m_centers.begin(); i != m_centers.end(); ++i)
		{
			vector<cv::Point> points(1);
			points[0].x = i->x;
			points[0].y = i->y;
			m_listCenters.push_back(points);
		}
	}

	else
	{
		vector<int> index(m_centers.size(), 1);//表示m_centers还剩哪些元素，已经被使用的元素置为0

		for (auto it = m_listCenters.begin(); it != m_listCenters.end();)
		{
			cv::Point pLast = it->at(it->size() - 1);//拿到列表当前向量的最后一个点
			int id;
			FindNearstPoint(pLast,id,index);

			if (id == -1)
			{
				it = m_listCenters.erase(it);
			}
			else
			{
				it->push_back(m_centers.at(id));
				++it;
			}
		}
		for (unsigned int i = 0; i < index.size();++i)
		{
			if (index[i] == 1)
			{
				vector<cv::Point> points(1);
				points[0].x = m_centers[i].x;
				points[0].y = m_centers[i].y;
				m_listCenters.push_back(points);
			}
		}
	}
}

void CBlob::DrawPaths(cv::Mat& _outputImg)
{
	cv::Mat outROI = _outputImg(cv::Rect(_outputImg.cols / 20, _outputImg.rows / 20, _outputImg.cols * 18 / 20, _outputImg.rows * 18 / 20));
	for (auto lisIt = m_listCenters.begin(); lisIt != m_listCenters.end(); ++lisIt)
	{
		for (unsigned int i = 0; i < lisIt->size();++i)
		{
			if (i>0)
			{
				cv::line(outROI, (*lisIt).at(i - 1), (*lisIt).at(i), cv::Scalar(0, 0, 255), 2, 8);
			}
		}
	}
}
