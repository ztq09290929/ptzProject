#include "blob2.h"

void CBlob::BlobDetecter(const cv::Mat& _binImg, cv::Mat& _outputImg)
{
	cv::Mat src = _binImg.clone();
	cv::Mat srcROI = src(cv::Rect(src.cols / 20, src.rows / 20, src.cols * 18 / 20, src.rows * 18 / 20));//�е�ͼ��ı�Ե5%
	cv::Mat outROI = _outputImg(cv::Rect(_outputImg.cols / 20, _outputImg.rows / 20, _outputImg.cols * 18 / 20, _outputImg.rows * 18 / 20));

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(srcROI, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	std::vector<std::vector<cv::Point>> goodContours;
	for (auto i = contours.begin(); i != contours.end(); ++i)
	{
		if (i->size() > 45)///��Ϊ�趨����ֵ���ų���С������
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
	//	if (m_centers.empty())cout << "m_centersΪ��" << endl;
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
		id = -1;
	}
}
void CBlob::FindNearstPointKal(cv::Point pLast, int& id, vector<int>& index)
{
	double dist = 0;
	double minDist = 10000;
	int count = 0;
	//	if (m_centers.empty())cout << "m_centersΪ��" << endl;
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
	if (minDist < 70)
	{
		index[id] = 0;
	}
	else
	{
		id = -1;
	}
}

void CBlob::ClassifyCenters()
{
	if (m_listCenters.empty())
	{
		for (auto i = m_centers.begin(); i != m_centers.end(); ++i)
		{
			ObjectAndKF obj1;
			obj1.m_vecCenters.push_back(cv::Point(i->x, i->y));
			obj1.Init();
			m_listCenters.push_back(obj1);
		}
	}

	else
	{
		vector<int> index(m_centers.size(), 1);//��ʾm_centers��ʣ��ЩԪ�أ��Ѿ���ʹ�õ�Ԫ����Ϊ0

		for (auto it = m_listCenters.begin(); it != m_listCenters.end();)
		{
			if (it->m_vecCenters.size() ==1)
			{
				cv::Point pLast = it->GetLastPoint();//�õ��б�ǰobj�����һ����
				int id;
				FindNearstPoint(pLast, id, index);

				if (id == -1)
				{
					it = m_listCenters.erase(it);
				}
				else
				{
					it->m_vecCenters.push_back(m_centers.at(id));
					it->Predict();
					it->Correct(1);
					++it;
				}
			}
			else
			{
				cv::Point prePoint = it->Predict();
				int id;
				FindNearstPointKal(prePoint, id, index);

				if (id == -1)
				{
					it = m_listCenters.erase(it);
				}
				else
				{
					it->m_vecCenters.push_back(m_centers.at(id));
					it->Correct(it->m_vecCenters.size()-1);
					++it;
				}
			}
		}
		for (unsigned int i = 0; i < index.size(); ++i)
		{
			if (index[i] == 1)
			{
				ObjectAndKF obj2;
				obj2.m_vecCenters.push_back(cv::Point(m_centers[i].x,m_centers[i].y));
				obj2.Init();
				m_listCenters.push_back(obj2);
			}
		}
	}
}

void CBlob::DrawPaths(cv::Mat& _outputImg)
{
	cv::Mat outROI = _outputImg(cv::Rect(_outputImg.cols / 20, _outputImg.rows / 20, _outputImg.cols * 18 / 20, _outputImg.rows * 18 / 20));
	for (auto lisIt = m_listCenters.begin(); lisIt != m_listCenters.end(); ++lisIt)
	{
		for (unsigned int i = 0; i < lisIt->m_vecCenters.size(); ++i)
		{
			if (i>0)
			{
				cv::line(outROI, lisIt->m_vecCenters.at(i - 1), lisIt->m_vecCenters.at(i), cv::Scalar(0, 0, 255), 2, 8);
			}
		}
	}
}



void ObjectAndKF::Init()
{
	KF = KalmanFilter(stateNum, measureNum, 0);
	state = Mat(stateNum, 1, CV_32FC1); //state(x,y,detaX,detaY)  
	processNoise = Mat(stateNum, 1, CV_32F);
	measurement = Mat::zeros(measureNum, 1, CV_32F);    //measurement(x,y)  

	randn(state, Scalar::all(0), Scalar::all(0.1));//�������һ������������0����׼��Ϊ0.1;  
	state.at<float>(0) = m_vecCenters.at(0).x;//������һ֡�ĳ�ʼλ������x��y
	state.at<float>(1) = m_vecCenters.at(0).y;

	KF.transitionMatrix = *(Mat_<float>(4, 4) <<
		1, 0, 1, 0,
		0, 1, 0, 1,
		0, 0, 1, 0,
		0, 0, 0, 1);//Ԫ�ص�����󣬰���;  
	//setIdentity: ���ŵĵ�λ�ԽǾ���;  
	//!< measurement matrix (H) �۲�ģ��  
	setIdentity(KF.measurementMatrix);

	//!< process noise covariance matrix (Q)  
	// wk �ǹ������������ٶ�����Ͼ�ֵΪ�㣬Э�������ΪQk(Q)�Ķ�Ԫ��̬�ֲ�;  
	setIdentity(KF.processNoiseCov, Scalar::all(1e-5));

	//!< measurement noise covariance matrix (R)  
	//vk �ǹ۲����������ֵΪ�㣬Э�������ΪRk,�ҷ�����̬�ֲ�;  
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));

	//!< priori error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q)*/  A����F: transitionMatrix  
	//Ԥ�����Э�������;  
	setIdentity(KF.errorCovPost, Scalar::all(1));

	//!< corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))  
	//initialize post state of kalman filter at random   
	randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));
	KF.statePost.at<float>(0) = m_vecCenters.at(0).x;//������һ֡�ĳ�ʼλ������x��y
	KF.statePost.at<float>(1) = m_vecCenters.at(0).y;
}

cv::Point ObjectAndKF::Predict()
{
	Point statePt = Point((int)KF.statePost.at<float>(0), (int)KF.statePost.at<float>(1));

	//2.kalman prediction     
	Mat prediction = KF.predict();
	Point predictPt = Point((int)prediction.at<float>(0), (int)prediction.at<float>(1));

	return predictPt;
}
void ObjectAndKF::Correct(int num)
{
	//3.update measurement  
	measurement.at<float>(0) = (float)m_vecCenters.at(num).x;
	measurement.at<float>(1) = (float)m_vecCenters.at(num).y;

	//4.update  
	KF.correct(measurement);
}