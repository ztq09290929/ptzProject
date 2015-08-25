#include "KeyPointMatch.h"

void KeyPointMatch::Set_trainImage(Mat scene)
{
	//cvtColor(trainImage, trainImage_gray, CV_BGR2GRAY);
	trainImage_gray = scene.clone();
	
#if SURF_USE
	SurfFeatureDetector featureDetector(hessian);
	featureDetector.detect(trainImage_gray, train_keyPoint);
	SurfDescriptorExtractor featureExtractor;
	featureExtractor.compute(trainImage_gray, train_keyPoint, trainDescriptor);
	//【3】创建基于FLANN的描述符匹配对象
	vector<Mat> train_desc_collection(1, trainDescriptor);
	matcher.add(train_desc_collection);
	matcher.train();
#else
	//【2】调用detect函数检测出特征关键点，保存在vector容器中
	OrbFeatureDetector featureDetector;
	featureDetector.detect(trainImage_gray, train_keyPoint);
	//【3】计算描述符（特征向量）
	OrbDescriptorExtractor featureExtractor;
	featureExtractor.compute(trainImage_gray, train_keyPoint, trainDescriptor);
#endif		

	
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
#else	
		//【2】调用detect函数检测出特征关键点，保存在vector容器中
		OrbFeatureDetector featureDetector;
		featureDetector.detect(testImage_gray, test_keyPoint);
		//【3】计算描述符（特征向量）
		OrbDescriptorExtractor featureExtractor;
		featureExtractor.compute(testImage_gray, test_keyPoint, testDescriptor);
		//【7】检测SIFT关键点并提取测试图像中的描述符
		//Mat testDescription;

		////【8】调用detect函数检测出特征关键点，保存在vector容器中
		//featureDetector.detect(testImage_gray, test_keyPoint);

		////【9】计算描述符
		//featureExtractor.compute(testImage_gray, test_keyPoint, testDescriptor);

		//【10】匹配和测试描述符，获取两个最邻近的描述符
		//【4】基于FLANN的描述符对象匹配
		flann::Index flannIndex(trainDescriptor, flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);

		Mat matchIndex(testDescriptor.rows, 2, CV_32SC1), matchDistance(testDescriptor.rows, 2, CV_32FC1);
		flannIndex.knnSearch(testDescriptor, matchIndex, matchDistance, 2, flann::SearchParams());//调用K邻近算法

		//【11】根据劳氏算法（Lowe's algorithm）选出优秀的匹配
		vector<DMatch> goodMatches;
		for (int i = 0; i < matchDistance.rows; i++)
		{
			if (matchDistance.at<float>(i, 0) < 0.6 * matchDistance.at<float>(i, 1))
			{
				DMatch dmatches(i, matchIndex.at<int>(i, 0), matchDistance.at<float>(i, 0));
				goodMatches.push_back(dmatches);
			}
		}
#endif

	

	//<6>绘制匹配点并显示窗口
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


