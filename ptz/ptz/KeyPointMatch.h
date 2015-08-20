#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <time.h>
#define SURF_USE 1//1的时候用surf，0的时候用orb
using namespace cv;
using namespace std;



class KeyPointMatch//必须先初始化void Set_trainImage_data();后才可以正常使用别的函数
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
	int hessian = 400;//surf特征提取的海森阈值
	std::vector<Point3f> TransformKeyPoint;
};
