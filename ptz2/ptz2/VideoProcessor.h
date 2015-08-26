#ifndef _VIDEOPROCESSOR_
#define  _VIDEOPROCESSOR_
#include <opencv2/opencv.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/stitching/stitcher.hpp"
#include <iostream>
#include <string>
#include <time.h>
#include "ViBe2.h"
#include "KeyPointMatch.h"
#include "blob.h"
//#include "opencv2/video/background_segm.hpp"
//#include "opencv2/legacy/blobtrack.hpp"
//#include "opencv2/legacy/legacy.hpp"
//#include <opencv2/imgproc/imgproc_c.h>

using namespace std;
using namespace cv;


class VideoProcessor
{
public:
inline	VideoProcessor():m_delay(1)//���캯������ʼ���б���ֵ
	{
//		blob = cvCreateBlobTrackerAuto1();
	}
//inline	~VideoProcessor(){ delete blob; }

	void Init(std::string filename);//��ȡȫ��ͼ�񣬻�ȡȫ��ͼ�����������ȫ��ͼ����ViBe��ʼ����

	int SetInput(std::string filename);//����������Ƶ�ļ����ƣ�������Ƶ
	void SetDelay(int d);//�����ӳ�

	void DisplayInput(std::string wn);//�������봰��
	void DisplayOutputFront(std::string wn);//�������ǰ������
	void DisplayOutputBack(std::string wn);//���������������

	void Run();//��ʼ����ѭ��

private:
	cv::VideoCapture m_capture;//�򿪵���Ƶ

	std::string m_windowNameInput;//���봰������
	std::string m_windowNameOutputFront;//���ǰ����������
	std::string m_windowNameOutputBack;//���������������

	cv::Mat m_frame;//�洢��ǰ��ȡ��һ֡ͼ��
	cv::Mat m_foreImage;//ǰ��ͼ��
	cv::Mat m_backImage;//����ͼ��
	cv::Mat m_pano;//ȫ��ͼ��
	std::vector<cv::Point3f> m_curFrame;//����任��ȫ������ͼ֮��ĵ�ǰ֡�����е㣬����Point3f�����������ֱ�Ϊx,y,gray

	ViBe_BGS vibe_bgs;//ViBe�㷨��
	KeyPointMatch keyPointMatch;//ͼ����׼��
//	CvBlobTrackerAuto* blob;
	CBlob cblob;//Ŀ�������

	int m_delay;//ÿ��֮֡���ʱ����

};

#endif