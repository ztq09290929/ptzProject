#include "VideoProcessor.h"

using namespace cv;

void VideoProcessor::Init(std::string filename)
{
	///生成全景图
	cv::VideoCapture cap;
	cap.open(filename);
	if (!cap.isOpened())
	{
		std::cout << "全景图-视频打开失败！" << std::endl;
		return ;
	}
	cv::Mat frame;//存储当前帧
	std::vector<cv::Mat> backImages;//存储背景的5个子图
	int count = 0;

	while (true)
	{
		++count;
		cap.read(frame);
		if (frame.empty())
		{
			std::cout << "全景图-图像读取失败" << std::endl;
			return ;
		}

		if (count > 600 && ( count==650||count % 100 == 1 || count == 1152) && count < 1200)
		{
			backImages.push_back(frame.clone());
		}
		if (count >= 1200)
		{
			break;
		}
	}
	std::cout << "子图个数：" << backImages.size() << "--- 正在生成全景图..." << std::endl;
	Stitcher stitcher = Stitcher::createDefault(false);
	Stitcher::Status status = stitcher.stitch(backImages, m_pano);
	if (status != Stitcher::OK)
	{
		cout << "全景图-Can't stitch images, error code=" << int(status) << endl;
		return ;
	}
	cv::cvtColor(m_pano, m_pano, cv::COLOR_BGR2GRAY);
	std::cout << "全景图生成成功！" << "全景图尺寸:" << m_pano.cols << "*" << m_pano.rows<<std::endl;
	cv::imshow("pano", m_pano);
	cv::waitKey();
	cv::destroyAllWindows();

	///提取全景图特征
	std::cout << "提取全景图特征成功！" << std::endl;
	keyPointMatch.Set_trainImage(m_pano);

	///初始化ViBe背景模型
	vibe_bgs.init(m_pano,frame);
	vibe_bgs.processFirstFrame(m_pano);
	std::cout << "初始化ViBe背景模型成功！" << std::endl;
	return ;
}

int VideoProcessor::SetInput(std::string filename)
{
	m_capture.release();

	m_capture.open(filename);
	if (!m_capture.isOpened())
	{
		std::cout << "视频打开失败！" << std::endl;
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
		//计时
		clock_t t1, t2;

		cv::imshow(m_windowNameInput, m_frame);
			
		t1 = clock();

		cv::cvtColor(m_frame, m_frame, cv::COLOR_BGR2GRAY);
		cv::GaussianBlur(m_frame, m_frame, cv::Size(3, 3),0,0);

		keyPointMatch.Set_testImage(m_frame);
		keyPointMatch.Get_H();
		m_curFrame = keyPointMatch.Get_TransformKeyPoint();
		t2 = clock();
		std::cout << "图像配准时间：" << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;

		t1 = clock();
		vibe_bgs.testAndUpdate(m_curFrame);
		m_backImage = vibe_bgs.getMask();//获取全景前景滑窗
		m_foreImage = vibe_bgs.getFore();//获取前景
		cv::medianBlur(m_foreImage, m_foreImage, 3);
		cv::Mat element = cv::getStructuringElement(MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
		cv::morphologyEx(m_foreImage, m_foreImage, MORPH_OPEN, element, cv::Point(-1, -1), 1);
		

		t2 = clock();
		std::cout << "ViBe所用时间：" << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;

		cv::imshow(m_windowNameOutputBack, m_backImage);
		cv::imshow(m_windowNameOutputFront, m_foreImage);
		if (cv::waitKey(m_delay) == 27)break;

	}
}

