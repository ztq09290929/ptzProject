#include "blob2.h"

void CBlob::BlobDetecter(const cv::Mat& _binImg, cv::Mat& _outputImg)
{
	cv::Mat src = _binImg.clone();
	cv::Mat srcROI = src(cv::Rect(src.cols / 20, src.rows / 20, src.cols * 18 / 20, src.rows * 18 / 20));//切掉图像的边缘5%
	cv::Mat outROI = _outputImg(cv::Rect(_outputImg.cols / 20, _outputImg.rows / 20, _outputImg.cols * 18 / 20, _outputImg.rows * 18 / 20));//为了坐标方便，输出图像也要提取ROI

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

	m_centers.clear();//处理每一帧图像，都要清空之前的点
	std::vector<cv::Rect> rects;
	for (auto i = goodContours.begin(); i != goodContours.end(); ++i)
	{
		cv::Rect temRect = cv::boundingRect(*i);//求轮廓的外接矩形
		rects.push_back(temRect);
		cv::Point center;
		int xSum(0), ySum(0);
		for (auto j = i->begin(); j != i->end(); ++j)
		{
			xSum += j->x;
			ySum += j->y;
		}
		center.x = (int)(xSum / i->size());//轮廓的中心点
		center.y = (int)(ySum / i->size());
		//center.x = temRect.x + temRect.width / 2;//矩形的中心点
		//center.y = temRect.y + temRect.height / 2;
		m_centers.push_back(center);//存入外接矩形的中心点
	}


	for (auto i = rects.begin(); i != rects.end(); ++i)
	{
		cv::rectangle(outROI, *i, cv::Scalar(0, 255, 0), 2, 8);//画出各物体的外接矩形
	}

}

void CBlob::FindNearstPoint(cv::Point pLast, int& id, vector<int>& index)
{
	double dist = 0;
	double minDist = 10000;
	int count = 0;
	//	if (m_centers.empty())cout << "m_centers为空" << endl;
	for (auto i = m_centers.begin(); i != m_centers.end(); ++i)//首先遍历当前帧矩形中心点集，提取与输入点的最小距离
	{
		dist = sqrt((pLast.x - (i->x))*(pLast.x - (i->x)) + (pLast.y - (i->y))*(pLast.y - (i->y)));

		if (dist < minDist)
		{
			minDist = dist;
			id = count;
		}
		++count;
	}
	if (minDist < 100)//如果输入点与中心点集最小距离满足要求，则将相应位置的标记置为0，表示该点已被用过
	{
		index[id] = 0;
	}
	else//如果距离太大，或者根本没有矩形中心点，即当前帧没有物体
	{
		id = -1;
	}
}
void CBlob::FindNearstPointKal(cv::Point pLast, int& id, vector<int>& index)
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
	if (minDist < 78)//与上一函数功能相同，只是输入的点为卡尔曼滤波的预测点，在该预测点附近查找
	{
		index[id] = 0;
	}
	else
	{
		id = -1;
	}
}

void CBlob::ClassifyCenters(Mat& _outputImg)
{
	cv::Mat outROI = _outputImg(cv::Rect(_outputImg.cols / 20, _outputImg.rows / 20, _outputImg.cols * 18 / 20, _outputImg.rows * 18 / 20));
	if (m_listCenters.empty())//如果存储物体的链表为空，首先用当前帧的矩形中心点集去初始化各个物体
	{
		for (auto i = m_centers.begin(); i != m_centers.end(); ++i)//每一点都是一个新的物体
		{
			ObjectAndKF obj1;
			obj1.m_vecCenters.push_back(cv::Point(i->x, i->y));
			obj1.Init();//存入该物体的第一个中心点位置后，初始化其卡尔曼滤波器
			m_listCenters.push_back(obj1);
		}
	}

	else//如果链表中有元素，则代表上一帧有物体
	{
		vector<int> index(m_centers.size(), 1);//表示m_centers还剩哪些元素，已经被使用的元素置为0，未被匹配的元素为1

		for (auto it = m_listCenters.begin(); it != m_listCenters.end();)//遍历链表中的每一个物体对象
		{
			if (it->m_vecCenters.size() ==1)//如果该物体对象只保存了一个位置点，不能使用卡尔曼滤波
			{
				cv::Point pLast = it->GetLastPoint();//拿到列表当前obj的最后一个点
				int id;
				FindNearstPoint(pLast, id, index);//寻找和这一点最近的当前帧中的点，认为是下一点

				if (id == -1)//如果没找到，则认为物体已经消失，则删除链表中的该项
				{
					it = m_listCenters.erase(it);
				}
				else//如果找到了，则将第二点的坐标存入向量，并且用其修正卡尔曼滤波器
				{
					it->m_vecCenters.push_back(m_centers.at(id));
					it->Predict();
					it->Correct(1);
					++it;
				}
			}
			else//如果该物体对象存着2个以上的位置点，则可以使用卡尔曼滤波器
			{
				cv::Point prePoint = it->Predict();//预估出当前帧这一点的位置

				cv::circle(outROI, prePoint, 5, cv::Scalar(0, 255, 255), 3, 8);

				int id;
				FindNearstPointKal(prePoint, id, index);//在预估的位置附近寻找真实的点的位置

				if (id == -1)//如果没找到，则认为物体已经消失，则删除链表中的该项
				{
					it = m_listCenters.erase(it);
				}
				else//如果找到了，则将这点的坐标存入向量，并且用其修正卡尔曼滤波器
				{
					it->m_vecCenters.push_back(m_centers.at(id));
					it->Correct(it->m_vecCenters.size()-1);
					++it;
				}
			}
		}
		for (unsigned int i = 0; i < index.size(); ++i)//对于中心点集中，未被置为0的点，认为他们是本帧中新出现的物体的中心点
		{
			if (index[i] == 1)
			{
				ObjectAndKF obj2;//则创建新的物体对象
				obj2.m_vecCenters.push_back(cv::Point(m_centers[i].x,m_centers[i].y));
				obj2.Init();
				m_listCenters.push_back(obj2);//将该对象存入链表
			}
		}
	}
}

void CBlob::DrawPaths(cv::Mat& _outputImg)
{
	//由于坐标原因，画图时也应取出ROI
	cv::Mat outROI = _outputImg(cv::Rect(_outputImg.cols / 20, _outputImg.rows / 20, _outputImg.cols * 18 / 20, _outputImg.rows * 18 / 20));
	for (auto lisIt = m_listCenters.begin(); lisIt != m_listCenters.end(); ++lisIt)//画出链表中的各点，每个物体的点各自连成直线
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

	randn(state, Scalar::all(0), Scalar::all(0.1));//随机生成一个矩阵，期望是0，标准差为0.1;  
	state.at<float>(0) = (float)m_vecCenters.at(0).x;//付给第一帧的初始位置坐标x和y
	state.at<float>(1) = (float)m_vecCenters.at(0).y;

	KF.transitionMatrix = *(Mat_<float>(4, 4) <<
		1, 0, 1, 0,
		0, 1, 0, 1,
		0, 0, 1, 0,
		0, 0, 0, 1);//元素导入矩阵，按行;  
	//setIdentity: 缩放的单位对角矩阵;  
	//!< measurement matrix (H) 观测模型  
	setIdentity(KF.measurementMatrix);

	//!< process noise covariance matrix (Q)  
	// wk 是过程噪声，并假定其符合均值为零，协方差矩阵为Qk(Q)的多元正态分布;  
	setIdentity(KF.processNoiseCov, Scalar::all(1e-5));

	//!< measurement noise covariance matrix (R)  
	//vk 是观测噪声，其均值为零，协方差矩阵为Rk,且服从正态分布;  
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));

	//!< priori error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q)*/  A代表F: transitionMatrix  
	//预测估计协方差矩阵;  
	setIdentity(KF.errorCovPost, Scalar::all(1));

	//!< corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))  
	//initialize post state of kalman filter at random   
	randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));
	KF.statePost.at<float>(0) = (float)m_vecCenters.at(0).x;//付给第一帧的初始位置坐标x和y
	KF.statePost.at<float>(1) = (float)m_vecCenters.at(0).y;
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