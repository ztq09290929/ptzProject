#include "ViBe2.h"


using namespace std;
using namespace cv;

int c_xoff[9] = { 0, 0, 0 ,- 1, -1, -1, 1, 1, 1 };  //x的邻居点
int c_yoff[9] = { 0, -1, 1, 0, -1, 1, 0, -1, 1 };  //y的邻居点

ViBe_BGS::ViBe_BGS(void)
{

}
ViBe_BGS::~ViBe_BGS(void)
{
	//释放三维数组
	for (int i = 0; i < imgRows; ++i)
	{
		for (int j = 0; j < imgCols; ++j)
		{
			delete[] samples[i][j];
		}
	}
	for (int i = 0; i < imgRows; ++i)
	{
		delete[] samples[i];
	}
	delete[] samples;
}

/**************** Assign space and init ***************************/
void ViBe_BGS::init(const Mat& _image, Mat frame)
{
	//首先记录分配动态数组的行列数, 第三维是NUM_SAMPLES + 1，在析构时用来释放内存
	imgRows = _image.rows;
	imgCols = _image.cols;
	//动态分配三维数组，samples[][][NUM_SAMPLES]存储前景被连续检测的次数
	samples = new unsigned char **[_image.rows];
	for (int i = 0; i < _image.rows; i++)
	{
		samples[i] = new unsigned char *[_image.cols];
		for (int j = 0; j < _image.cols; j++)
		{
			samples[i][j] = new unsigned char[NUM_SAMPLES + 1];//最后一个存储前景计数器
			for (int k = 0; k < NUM_SAMPLES + 1; k++)
			{
				samples[i][j][k] = 0;
			}
		}
	}

	m_pano = _image.clone();//存储全景图
	m_mask = Mat::zeros(_image.size(), CV_8UC1);//大小和全景图相同
	m_fore = Mat::zeros(frame.size(), CV_8UC1);//大小和当前帧相同
	return;
}

/**************** Init model from first frame ********************/
void ViBe_BGS::processFirstFrame(const Mat& _image)
{
	RNG rng;
	int row, col;

	for (int i = 0; i < _image.rows; i++)
	{
		for (int j = 0; j < _image.cols; j++)
		{
			for (int k = 0; k < NUM_SAMPLES; k++)
			{
				// Random pick up NUM_SAMPLES pixel in neighbourhood to construct the model
				int random = rng.uniform(0, 9);

				row = i + c_yoff[random];
				if (row < 0)
					row = 0;
				if (row >= _image.rows)
					row = _image.rows - 1;

				random = rng.uniform(0, 9);
				col = j + c_xoff[random];
				if (col < 0)
					col = 0;
				if (col >= _image.cols)
					col = _image.cols - 1;

				samples[i][j][k] = _image.at<uchar>(row, col);
			}
		}
	}
}

uchar ViBe_BGS::getNbhdPoints(float row, float col)
{
	int leftX, rightX, topY, botY;
	leftX = ceil(col - RADIUS_NBHD) < 0 ? 0 : ceil(col - RADIUS_NBHD);//获取圆形区域的外接矩形
	rightX = floor(col + RADIUS_NBHD)>(imgCols - 1) ? (imgCols - 1) : floor(col + RADIUS_NBHD);
	topY = ceil(row - RADIUS_NBHD) < 0 ? 0 : ceil(row - RADIUS_NBHD);
	botY = floor(row + RADIUS_NBHD)>(imgRows - 1) ? (imgRows - 1) : floor(row + RADIUS_NBHD);
	int count = 0; 
	for (int i = topY; i <= botY; ++i)//遍历外接矩形区域，找到圆形中的点
	{
		for (int j = leftX; j <= rightX; ++j)
		{
			if (((i - row)*(i - row) + (j - col)*(j - col)) <= RADIUS_NBHD*RADIUS_NBHD)//如果该点到中心的距离不大于半径，则在园内
			{
				//cv::Point2i temp;
				//temp.x = j;
				//temp.y = i;
				if (count < ((RADIUS_NBHD * 2 - 1)*(RADIUS_NBHD * 2 - 1) + 4 + 2))
				{
					m_NbhdPoints[count][0] = j;
					m_NbhdPoints[count][1] = i;
					++count;
				}
			}
		}
	}
	return (uchar)count;//count为属于当前圆的点的个数
}

/**************** Test a new frame and update model ********************/
void ViBe_BGS::testAndUpdate(const std::vector<cv::Point3f>& _image)
{
	m_mask = m_pano.clone();//将存储的灰度全景图复制给m_mask
	RNG rng;
	for (unsigned int i = 0; i < _image.size(); i++)//遍历所有传入的像素点
	{
		float xfCol = _image[i].x ;//得到这一点的x和y坐标以及灰度值
		float yfRow = _image[i].y ;
		int xCol = (int)(xfCol + 0.5);//得到这一点四舍五入之后的x和y坐标以及灰度值
		int yRow = (int)(yfRow + 0.5);
		uchar gray = (uchar)_image[i].z;
		if (0 <= xfCol && xfCol <= (imgCols - 1) && 0 <= yfRow && yfRow <= (imgRows - 1))//如果该点在背景之内，继续处理
		{

			int votes = 0;
			unsigned int j = 0;
			uchar num = getNbhdPoints(yfRow, xfCol);//得到在该点圆形邻域内的点，以及数量

			while (votes < VOTES && j < num)//遍历圆形邻域的每一点，如果有一个点认为该像素是背景，则它是背景
			{
				int matches(0), count(0);
				int dist;
				int row, col;
				row = m_NbhdPoints[j][1];
				col = m_NbhdPoints[j][0];

				while (matches < MIN_MATCHES && count < NUM_SAMPLES)//与20个样本值比较
				{
					dist = abs(samples[row][col][count] - gray);//采用指针遍历方法更快
					if (dist < RADIUS)
						matches++;
					count++;
				}
				if (matches >= MIN_MATCHES)
				{
					++votes;
				}
				++j;
			}

			if (votes >= VOTES)//如果是背景
			{
				// It is a background pixel
				samples[yRow][xCol][NUM_SAMPLES] = 0;//前景计数器归零

				// Set background pixel to 0
				m_mask.at<uchar>(yRow, xCol) = 0;
				m_fore.at<uchar>((int)i / m_fore.cols, (int)i%m_fore.cols) = 0;//将输出图像的相应位置置为黑色0
				// 如果一个像素是背景点，那么它有 1 / defaultSubsamplingFactor 的概率去更新自己的模型样本值
				int random = rng.uniform(0, SUBSAMPLE_FACTOR);
				if (random == 0)
				{
					random = rng.uniform(0, NUM_SAMPLES);
					samples[yRow][xCol][random] = gray;//此处更新背景还需改进
				}

				// 同时也有 1 / defaultSubsamplingFactor 的概率去更新它的邻居点的模型样本值
				random = rng.uniform(0, SUBSAMPLE_FACTOR);
				if (random == 0)
				{
					int row, col;
					random = rng.uniform(0, 9);
					row = yRow + c_yoff[random];
					if (row < 0)
						row = 0;
					if (row >= imgRows)
						row = imgRows - 1;

					random = rng.uniform(0, 9);
					col = xCol + c_xoff[random];
					if (col < 0)
						col = 0;
					if (col >= imgCols)
						col = imgCols - 1;

					random = rng.uniform(0, NUM_SAMPLES);
					samples[row][col][random] = gray;//此处更新背景还需改进
				}
			}

			if (votes < VOTES)//如果是前景
			{
				// It is a foreground pixel
				++samples[yRow][xCol][NUM_SAMPLES];//前景计数器加1

				// Set background pixel to 255
				m_mask.at<uchar>(yRow, xCol) = 255;
				m_fore.at<uchar>((int)i / m_fore.cols, (int)i%m_fore.cols) = 255;//将输出图像的相应位置置为白色255
				//如果某个像素点连续N次被检测为前景，则认为一块静止区域被误判为运动，将其更新为背景点
				if (samples[yRow][xCol][NUM_SAMPLES] > 40)
				{
					int random = rng.uniform(0, SUBSAMPLE_FACTOR);
					if (random == 0)
					{
						random = rng.uniform(0, NUM_SAMPLES);
						samples[yRow][xCol][random] = gray;//此处更新背景还需改进

					}
					// 同时也有 1 / defaultSubsamplingFactor 的概率去更新它的邻居点的模型样本值
					random = rng.uniform(0, SUBSAMPLE_FACTOR);
					if (random == 0)
					{
						int row, col;
						random = rng.uniform(0, 9);
						row = yRow + c_yoff[random];
						if (row < 0)
							row = 0;
						if (row >= imgRows)
							row = imgRows - 1;

						random = rng.uniform(0, 9);
						col = xCol + c_xoff[random];
						if (col < 0)
							col = 0;
						if (col >= imgCols)
							col = imgCols - 1;

						random = rng.uniform(0, NUM_SAMPLES);
						samples[row][col][random] = gray;//此处更新背景还需改进
					}
				}
			}
		}
	}
}