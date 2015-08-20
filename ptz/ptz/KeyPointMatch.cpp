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

	//��3����������FLANN��������ƥ�����
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

	//<1>��������
	//resize(testImage, testImage, Size(360, 240));

	//<2>ת��ͼ�񵽻Ҷ�
	testImage_gray = testImage.clone();
	//cvtColor(testImage, testImage_gray, CV_BGR2GRAY);

	//<3>���S�ؼ��㡢��ȡ����ͼ��������,���Method = 0,Surf�ؼ��㡢��ȡѵ��ͼ��������.Method=1����ORB�����㣬����1Ĭ����surf
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

	//<4>ƥ��ѵ���Ͳ���������
	vector<vector<DMatch> > matches;
	matcher.knnMatch(testDescriptor, matches, 2);

	// <5>���������㷨��Lowe's algorithm�����õ������ƥ���
	vector<DMatch> goodMatches;
	for (unsigned int i = 0; i < matches.size(); i++)
	{
		if (matches[i][0].distance < 0.8 * matches[i][1].distance)///��ֵ��Ҫ������findHomography()���ܻ�����쳣
			goodMatches.push_back(matches[i][0]);
	}

	////<6>����ƥ��㲢��ʾ����
	//Mat dstImage;
	//drawMatches(testImage, test_keyPoint, trainImage, train_keyPoint, goodMatches, dstImage);
	//imshow("ƥ�䴰��", dstImage);

	//���������ֲ�����
	vector<Point2f> obj;
	vector<Point2f> scene;

	//��ƥ��ɹ���ƥ����л�ȡ�ؼ���
	for (unsigned int i = 0; i < goodMatches.size(); i++)
	{
		obj.push_back(test_keyPoint[goodMatches[i].queryIdx].pt);
		scene.push_back(train_keyPoint[goodMatches[i].trainIdx].pt);
	}

	 H = findHomography(obj, scene, CV_RANSAC);//����͸�ӱ任 

	 ////�Ӵ���ͼƬ�л�ȡ�ǵ�
	 //vector<Point2f> obj_corners(4);
	 //obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(testImage_gray.cols, 0);
	 //obj_corners[2] = cvPoint(testImage_gray.cols, testImage_gray.rows); obj_corners[3] = cvPoint(0, testImage_gray.rows);
	 //vector<Point2f> scene_corners(4);
	 ////����͸�ӱ任
	 //perspectiveTransform(obj_corners, scene_corners, H);
	 ////���Ƴ��ǵ�֮���ֱ��
	 //line(trainImage_gray, scene_corners[0], scene_corners[1], Scalar(255, 0, 123), 4);
	 //line(trainImage_gray, scene_corners[1], scene_corners[2], Scalar(255, 0, 123), 4);
	 //line(trainImage_gray, scene_corners[2], scene_corners[3], Scalar(255, 0, 123), 4);
	 //line(trainImage_gray, scene_corners[3], scene_corners[0], Scalar(255, 0, 123), 4);
	 ////��ʾ���ս��
	 //imshow("ceshiЧ��ͼ", trainImage_gray);
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
//	//��ʼ��
//	test.Set_trainImage(trainImage);
//	//test.Set_trainImage_data();
//	
//	
//	test.Set_testImage(testImage);
//	Mat H = test.Get_H();
//
//	//������
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
//	imshow("���Ծ���ת��", trainImage_gray);
//	t1 = clock();
//	std::cout << "��ʱ" << (double)(t1 - t0)/CLOCKS_PER_SEC << std::endl;
//	waitKey(0);
//	return 0;
//	
//}


//int main()
//{
//	//��0���ı�console������ɫ
//	system("color 6F");
//
//
//
//	//��1������ͼ����ʾ��ת��Ϊ�Ҷ�ͼ
//	Mat trainImage = imread("pano.jpg"), trainImage_gray;
//	imshow("ԭʼͼ", trainImage);
//	cvtColor(trainImage, trainImage_gray, CV_BGR2GRAY);
//
//	//��2�����Surf�ؼ��㡢��ȡѵ��ͼ��������
//	vector<KeyPoint> train_keyPoint;
//	Mat trainDescriptor;
//	SurfFeatureDetector featureDetector(1000);
//	featureDetector.detect(trainImage_gray, train_keyPoint);
//	SurfDescriptorExtractor featureExtractor;
//	featureExtractor.compute(trainImage_gray, train_keyPoint, trainDescriptor);
//
//	//��3����������FLANN��������ƥ�����
//	FlannBasedMatcher matcher;
//	vector<Mat> train_desc_collection(1, trainDescriptor);
//	matcher.add(train_desc_collection);
//	matcher.train();
//
//	
//		//<1>��������
//		int64 time0 = getTickCount();
//		Mat testImage, testImage_gray;
//		testImage = imread("2.jpg");//�õ���ƥ��testImage
//		resize(testImage, testImage, Size(360, 240));
//
//		//<2>ת��ͼ�񵽻Ҷ�
//		cvtColor(testImage, testImage_gray, CV_BGR2GRAY);
//
//		//<3>���S�ؼ��㡢��ȡ����ͼ��������
//		vector<KeyPoint> test_keyPoint;
//		Mat testDescriptor;
//		featureDetector.detect(testImage_gray, test_keyPoint);
//		featureExtractor.compute(testImage_gray, test_keyPoint, testDescriptor);
//
//		//<4>ƥ��ѵ���Ͳ���������
//		vector<vector<DMatch> > matches;
//		matcher.knnMatch(testDescriptor, matches, 2);
//
//		// <5>���������㷨��Lowe's algorithm�����õ������ƥ���
//		vector<DMatch> goodMatches;
//		for (unsigned int i = 0; i < matches.size(); i++)
//		{
//			if (matches[i][0].distance < 0.6 * matches[i][1].distance)
//				goodMatches.push_back(matches[i][0]);
//		}
//
//		//<6>����ƥ��㲢��ʾ����
//		Mat dstImage;
//		drawMatches(testImage, test_keyPoint, trainImage, train_keyPoint, goodMatches, dstImage);
//		imshow("ƥ�䴰��", dstImage);
//
//		//<7>���֡����Ϣ
//		cout << "��ǰ֡��Ϊ��" << getTickFrequency() / (getTickCount() - time0) << endl;
//	//}
//
//		//���������ֲ�����
//		vector<Point2f> obj;
//		vector<Point2f> scene;
//
//		//��ƥ��ɹ���ƥ����л�ȡ�ؼ���
//		for (unsigned int i = 0; i < goodMatches.size(); i++)
//		{
//			obj.push_back(test_keyPoint[goodMatches[i].queryIdx].pt);
//			scene.push_back(train_keyPoint[goodMatches[i].trainIdx].pt);
//		}
//
//		Mat H = findHomography(obj, scene, CV_RANSAC);//����͸�ӱ任 
//
// 		waitKey(0);
//	return 0;
//}
