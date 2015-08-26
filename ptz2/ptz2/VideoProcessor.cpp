#include "VideoProcessor.h"

using namespace cv;

void VideoProcessor::Init(std::string filename)
{
	///����ȫ��ͼ
	cv::VideoCapture cap;
	cap.open(filename);
	if (!cap.isOpened())
	{
		std::cout << "ȫ��ͼ-��Ƶ��ʧ�ܣ�" << std::endl;
		return ;
	}
	cv::Mat frame;//�洢��ǰ֡
	std::vector<cv::Mat> backImages;//�洢��������ͼ
	int count = 0;
	while (true)
	{
		++count;
		cap.read(frame);
		if (frame.empty())
		{
			std::cout << "ȫ��ͼ-ͼ���ȡʧ��" << std::endl;
			return;
		}

		if (count > 600 && ( count==650||count % 100 == 1 || count == 1152) && count < 1200)
		//if (count==5||count==370||count==400||count==445||count==490||count==540||count==600)
		{
			backImages.push_back(frame.clone());
		}
		if (count >= 1200)
		{
			break;
		}
	}
	std::cout << "��ͼ������" << backImages.size() << "--- ��������ȫ��ͼ..." << std::endl;
	Stitcher stitcher = Stitcher::createDefault(false);
	Stitcher::Status status = stitcher.stitch(backImages, m_pano);
	if (status != Stitcher::OK)
	{
		cout << "ȫ��ͼ-Can't stitch images, error code=" << int(status) << endl;
		return ;
	}
	cv::cvtColor(m_pano, m_pano, cv::COLOR_BGR2GRAY);
	std::cout << "ȫ��ͼ���ɳɹ���" << "ȫ��ͼ�ߴ�:" << m_pano.cols << "*" << m_pano.rows<<std::endl;
	cv::imshow("pano", m_pano);
	cv::waitKey();
	cv::destroyAllWindows();

	///��ȡȫ��ͼ����
	keyPointMatch.Set_trainImage(m_pano);
	std::cout << "��ȡȫ��ͼ�����ɹ���" << std::endl;

	///��ʼ��ViBe����ģ��
	vibe_bgs.init(m_pano,frame);
	vibe_bgs.processFirstFrame(m_pano);
	std::cout << "��ʼ��ViBe����ģ�ͳɹ���" << std::endl;


	return ;
}

int VideoProcessor::SetInput(std::string filename)
{
	m_capture.release();
	m_capture.open(filename);
	if (!m_capture.isOpened())
	{
		std::cout << "��Ƶ��ʧ�ܣ�" << std::endl;
		return -1;
	}
	return 0;
}

void VideoProcessor::DisplayInput(std::string wn)
{
	m_windowNameInput = wn;
	namedWindow(m_windowNameInput);
}
void VideoProcessor::DisplayOutputFront(std::string wn)
{
	m_windowNameOutputFront = wn;
	namedWindow(m_windowNameOutputFront);
}
void VideoProcessor::DisplayOutputBack(std::string wn)
{
	m_windowNameOutputBack = wn;
	namedWindow(m_windowNameOutputBack);
}

void VideoProcessor::SetDelay(int d)
{
	m_delay = d;
}

void VideoProcessor::Run()
{
	while (m_capture.read(m_frame))
	{
		//��ʱ
		clock_t t1, t2;

		cv::Mat colorImg = m_frame.clone();
		//cv::imshow(m_windowNameInput, colorImg);
			
		t1 = clock();

		cv::cvtColor(m_frame, m_frame, cv::COLOR_BGR2GRAY);
		cv::GaussianBlur(m_frame, m_frame, cv::Size(3, 3),0,0);

		keyPointMatch.Set_testImage(m_frame);
		keyPointMatch.Get_H();
		m_curFrame = keyPointMatch.Get_TransformKeyPoint();
		t2 = clock();
		std::cout << "ͼ����׼ʱ�䣺" << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;

		t1 = clock();
		vibe_bgs.testAndUpdate(m_curFrame);
		m_backImage = vibe_bgs.getMask();//��ȡȫ��ǰ������
		m_foreImage = vibe_bgs.getFore();//��ȡǰ��

		//cv::medianBlur(m_foreImage, m_foreImage, 3);
		cv::Mat element1 = cv::getStructuringElement(MORPH_RECT, cv::Size(5, 5), cv::Point(-1, -1));
		cv::Mat element2 = cv::getStructuringElement(MORPH_RECT, cv::Size(5, 5), cv::Point(-1, -1));	
		//cv::morphologyEx(m_foreImage, m_foreImage, MORPH_CLOSE, element1, cv::Point(-1, -1), 3);
		cv::dilate(m_foreImage, m_foreImage, element1, cv::Point(-1, -1), 3);
		cv::erode(m_foreImage, m_foreImage, element2, cv::Point(-1, -1), 5);

		t2 = clock();
		std::cout << "ViBe����ʱ�䣺" << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;

		//cv::Mat m_foreImageCopy = m_foreImage.clone();
		//cv::Ptr<IplImage> pMask = cvCreateImage(cvSize(m_foreImageCopy.cols, m_foreImageCopy.rows), 8, 1);
		//pMask->imageData = (char *)m_foreImageCopy.data;
		//cv::Mat m_frameCopy = m_frame.clone();
		//cv::Ptr<IplImage> pImg = cvCreateImage(cvSize(m_frameCopy.cols, m_frameCopy.rows), 8, 1);
		//pImg->imageData = (char *)m_frameCopy.data;
		//blob->Process(pImg, pMask);
		//cout << "�ſ�����" << blob->GetBlobNum() << endl;
		//cv:Mat img = Mat(pImg);
		cblob.BlobDetecter(m_foreImage, colorImg);
		cblob.ClassifyCenters();
		cblob.DrawPaths(colorImg);

		cv::imshow(m_windowNameInput, colorImg);
		cv::imshow(m_windowNameOutputBack, m_backImage);
		cv::imshow(m_windowNameOutputFront, m_foreImage);
		if (cv::waitKey(m_delay) == 27)break;
	}
}

