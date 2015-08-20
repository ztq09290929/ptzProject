#include "KeyPointMatch.h"

void KeyPointMatch::Set_trainImage(Mat scene)
{
	//cvtColor(trainImage, trainImage_gray, CV_BGR2GRAY);
	trainImage_gray = scene.clone();
	Mat trainDescriptor;
#if SURF_USE
	SurfFeatureDetector featureDetector(hessian);
	featureDetector.detect(trainImage_gray, train_keyPoint);
	SurfDescriptorExtractor featureExtractor;
	featureExtractor.compute(trainImage_gray, train_keyPoint, trainDescriptor);
#else
	OrbFeatureDetector featureDetector;
	featureDetector.detect(trainImage_gray, train_keyPoint);
	OrbDescriptorExtractor featureExtractor;
	featureExtractor.compute(trainImage_gray, train_keyPoint, trainDescriptor);
#endif	

	//【3】创建基于FLANN的描述符匹配对象
	vector<Mat> train_desc_collection(1, trainDescriptor);
	matcher.add(train_desc_collection);
	matcher.train();
}
void KeyPointMatch::Set_testImage(Mat obj)
{
	testImage = obj;
}
Mat KeyPointMatch::Get_H()
{

	//<1>参数设置
	//resize(testImage, testImage, Size(360, 240));

	//<2>转化图像到灰度
	testImage_gray = testImage.clone();
	//cvtColor(testImage, testImage_gray, CV_BGR2GRAY);

	//<3>检测S关键点、提取测试图像描述符,如果Method = 0,Surf关键点、提取训练图像描述符.Method=1就用ORB特征点，不是1默认用surf
	vector<KeyPoint> test_keyPoint;
	Mat testDescriptor;
#if SURF_USE
		SurfFeatureDetector featureDetector(hessian);
		featureDetector.detect(testImage_gray, test_keyPoint);
		SurfDescriptorExtractor featureExtractor;
		featureExtractor.compute(testImage_gray, test_keyPoint, testDescriptor);
#else	
		OrbFeatureDetector featureDetector(hessian);
		featureDetector.detect(testImage_gray, test_keyPoint);
		OrbDescriptorExtractor featureExtractor;
		featureExtractor.compute(testImage_gray, test_keyPoint, testDescriptor);
#endif

	//<4>匹配训练和测试描述符
	vector<vector<DMatch> > matches;
	matcher.knnMatch(testDescriptor, matches, 2);

	// <5>根据劳氏算法（Lowe's algorithm），得到优秀的匹配点
	vector<DMatch> goodMatches;
	for (unsigned int i = 0; i < matches.size(); i++)
	{
		if (matches[i][0].distance < 0.8 * matches[i][1].distance)///阈值需要调整，findHomography()可能会产生异常
			goodMatches.push_back(matches[i][0]);
	}

	////<6>绘制匹配点并显示窗口
	//Mat dstImage;
	//drawMatches(testImage, test_keyPoint, trainImage, train_keyPoint, goodMatches, dstImage);
	//imshow("匹配窗口", dstImage);

	//定义两个局部变量
	vector<Point2f> obj;
	vector<Point2f> scene;

	//从匹配成功的匹配对中获取关键点
	for (unsigned int i = 0; i < goodMatches.size(); i++)
	{
		obj.push_back(test_keyPoint[goodMatches[i].queryIdx].pt);
		scene.push_back(train_keyPoint[goodMatches[i].trainIdx].pt);
	}

	 H = findHomography(obj, scene, CV_RANSAC);//计算透视变换 

	 ////从待测图片中获取角点
	 //vector<Point2f> obj_corners(4);
	 //obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(testImage_gray.cols, 0);
	 //obj_corners[2] = cvPoint(testImage_gray.cols, testImage_gray.rows); obj_corners[3] = cvPoint(0, testImage_gray.rows);
	 //vector<Point2f> scene_corners(4);
	 ////进行透视变换
	 //perspectiveTransform(obj_corners, scene_corners, H);
	 ////绘制出角点之间的直线
	 //line(trainImage_gray, scene_corners[0], scene_corners[1], Scalar(255, 0, 123), 4);
	 //line(trainImage_gray, scene_corners[1], scene_corners[2], Scalar(255, 0, 123), 4);
	 //line(trainImage_gray, scene_corners[2], scene_corners[3], Scalar(255, 0, 123), 4);
	 //line(trainImage_gray, scene_corners[3], scene_corners[0], Scalar(255, 0, 123), 4);
	 ////显示最终结果
	 //imshow("ceshi效果图", trainImage_gray);
	 //waitKey(0);
	 return H;
	
}
std::vector<Point3f> KeyPointMatch::Get_TransformKeyPoint()
{
	TransformKeyPoint.clear();
	int col = testImage_gray.size().width;
	int row = testImage_gray.size().height;
	std::vector<Point2f> Vertex, TransVertex;
	Point2f tem;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			tem.x = j;
			tem.y = i;
			Vertex.push_back(tem);
		}
	}
	perspectiveTransform(Vertex, TransVertex, H);
	for (int i = 0; i < row; i++)
	{
		uchar* P_row = testImage_gray.ptr<uchar>(i);
		for (int j = 0; j < col; j++)
		{
			Point3f tempt;
			tempt.x = TransVertex[i*col + j].x;
			tempt.y = TransVertex[i*col + j].y;
			tempt.z = P_row[j];
			TransformKeyPoint.push_back(tempt);
		}
	}
	return TransformKeyPoint;

}

//int main()
//{
//	clock_t t0, t1;
//	t0 = clock();
//	KeyPointMatch test;
//	Mat trainImage = imread("pano.jpg");
//	Mat testImage = imread("2.jpg");
//	//初始化
//	test.Set_trainImage(trainImage);
//	//test.Set_trainImage_data();
//	
//	
//	test.Set_testImage(testImage);
//	Mat H = test.Get_H();
//
//	//测试用
//	Mat trainImage_gray;
//	cvtColor(trainImage, trainImage_gray, CV_BGR2GRAY);
//	std::vector<Point3f> TransImage = test.Get_TransformKeyPoint();
//	for (int i = 0; i < TransImage.size(); i++)
//	{
//		int x = TransImage[i].x;
//		int y = TransImage[i].y;
//		if (y<trainImage_gray.rows)
//		trainImage_gray.at<uchar>(y, x) = 255;
//	}
//	imshow("测试矩阵转化", trainImage_gray);
//	t1 = clock();
//	std::cout << "耗时" << (double)(t1 - t0)/CLOCKS_PER_SEC << std::endl;
//	waitKey(0);
//	return 0;
//	
//}


//int main()
//{
//	//【0】改变console字体颜色
//	system("color 6F");
//
//
//
//	//【1】载入图像、显示并转化为灰度图
//	Mat trainImage = imread("pano.jpg"), trainImage_gray;
//	imshow("原始图", trainImage);
//	cvtColor(trainImage, trainImage_gray, CV_BGR2GRAY);
//
//	//【2】检测Surf关键点、提取训练图像描述符
//	vector<KeyPoint> train_keyPoint;
//	Mat trainDescriptor;
//	SurfFeatureDetector featureDetector(1000);
//	featureDetector.detect(trainImage_gray, train_keyPoint);
//	SurfDescriptorExtractor featureExtractor;
//	featureExtractor.compute(trainImage_gray, train_keyPoint, trainDescriptor);
//
//	//【3】创建基于FLANN的描述符匹配对象
//	FlannBasedMatcher matcher;
//	vector<Mat> train_desc_collection(1, trainDescriptor);
//	matcher.add(train_desc_collection);
//	matcher.train();
//
//	
//		//<1>参数设置
//		int64 time0 = getTickCount();
//		Mat testImage, testImage_gray;
//		testImage = imread("2.jpg");//得到带匹配testImage
//		resize(testImage, testImage, Size(360, 240));
//
//		//<2>转化图像到灰度
//		cvtColor(testImage, testImage_gray, CV_BGR2GRAY);
//
//		//<3>检测S关键点、提取测试图像描述符
//		vector<KeyPoint> test_keyPoint;
//		Mat testDescriptor;
//		featureDetector.detect(testImage_gray, test_keyPoint);
//		featureExtractor.compute(testImage_gray, test_keyPoint, testDescriptor);
//
//		//<4>匹配训练和测试描述符
//		vector<vector<DMatch> > matches;
//		matcher.knnMatch(testDescriptor, matches, 2);
//
//		// <5>根据劳氏算法（Lowe's algorithm），得到优秀的匹配点
//		vector<DMatch> goodMatches;
//		for (unsigned int i = 0; i < matches.size(); i++)
//		{
//			if (matches[i][0].distance < 0.6 * matches[i][1].distance)
//				goodMatches.push_back(matches[i][0]);
//		}
//
//		//<6>绘制匹配点并显示窗口
//		Mat dstImage;
//		drawMatches(testImage, test_keyPoint, trainImage, train_keyPoint, goodMatches, dstImage);
//		imshow("匹配窗口", dstImage);
//
//		//<7>输出帧率信息
//		cout << "当前帧率为：" << getTickFrequency() / (getTickCount() - time0) << endl;
//	//}
//
//		//定义两个局部变量
//		vector<Point2f> obj;
//		vector<Point2f> scene;
//
//		//从匹配成功的匹配对中获取关键点
//		for (unsigned int i = 0; i < goodMatches.size(); i++)
//		{
//			obj.push_back(test_keyPoint[goodMatches[i].queryIdx].pt);
//			scene.push_back(train_keyPoint[goodMatches[i].trainIdx].pt);
//		}
//
//		Mat H = findHomography(obj, scene, CV_RANSAC);//计算透视变换 
//
// 		waitKey(0);
//	return 0;
//}
