#ifndef _KEYPOINTMATCH_
#define  _KEYPOINTMATCH_
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <time.h>
#define SURF_USE 1//1��ʱ����surf��0��ʱ����orb
using namespace cv;
using namespace std;



class KeyPointMatch//�����ȳ�ʼ��void Set_trainImage_data();��ſ�������ʹ�ñ�ĺ���
{
public:
	//KeyPointMatch(){};
	//~KeyPointMatch(){};
	void Set_trainImage(Mat scene);
	void Set_testImage(Mat obj);
	Mat Get_H();
	std::vector<Point3f> Get_TransformKeyPoint();
private:
	Mat trainImage;
	Mat testImage;
	Mat trainImage_gray;
	Mat testImage_gray;
	Mat H;
	FlannBasedMatcher matcher;
	vector<KeyPoint> train_keyPoint;
	int hessian = 400;//surf������ȡ�ĺ�ɭ��ֵ
	std::vector<Point3f> TransformKeyPoint;
	Mat trainDescriptor;//������������������
	//flann::Index flannIndex;
};
#endif