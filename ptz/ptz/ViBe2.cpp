#include "ViBe2.h"

using namespace std;
using namespace cv;

int c_xoff[9] = { -1, 0, 1, -1, 1, -1, 0, 1, 0 };  //x的邻居点
int c_yoff[9] = { -1, 0, 1, -1, 1, -1, 0, 1, 0 };  //y的邻居点

ViBe_BGS::ViBe_BGS(void)
{

}
ViBe_BGS::~ViBe_BGS(void)
{
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
void ViBe_BGS::init(const Mat _image)
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
	m_mask = Mat::zeros(_image.size(), CV_8UC1);
}

/**************** Init model from first frame ********************/
void ViBe_BGS::processFirstFrame(const Mat _image)
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

/**************** Test a new frame and update model ********************/
void ViBe_BGS::testAndUpdate(std::vector<cv::Point3f> _image)
{
	RNG rng;
	for (unsigned int i = 0; i < _image.size(); i++)
	{
		int xCol = (int)(_image[i].x + 0.5);//得到这一点四舍五入之后的x和y坐标以及灰度值
		int yRow = (int)(_image[i].y + 0.5);
		uchar gray = (uchar)_image[i].z;
		if (0 <= xCol && xCol < imgCols && 0 <= yRow && yRow < imgRows)//如果该点在背景之内，继续处理
		{
			int matches(0), count(0);
			int dist;

			while (matches < MIN_MATCHES && count < NUM_SAMPLES)
			{
				//dist = abs(samples[i][j][count] - _image.at<uchar>(i, j));
				dist = abs(samples[yRow][xCol][count] - gray);//采用指针遍历方法更快
				if (dist < RADIUS)
					matches++;
				count++;
			}

			if (matches >= MIN_MATCHES)
			{
				// It is a background pixel
				samples[yRow][xCol][NUM_SAMPLES] = 0;

				// Set background pixel to 0
				//m_mask.at<uchar>(i, j) = 0;

				m_mask.at<uchar>(yRow, xCol) = 0;

				// 如果一个像素是背景点，那么它有 1 / defaultSubsamplingFactor 的概率去更新自己的模型样本值
				int random = rng.uniform(0, SUBSAMPLE_FACTOR);
				if (random == 0)
				{
					random = rng.uniform(0, NUM_SAMPLES);
					//samples[i][j][random] = _image.at<uchar>(i, j);
					samples[yRow][xCol][random] = gray;//使用指针遍历更快

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
					//samples[row][col][random] = _image.at<uchar>(i, j);
					samples[row][col][random] = gray;
				}
			}

			if (matches < MIN_MATCHES)
			{
				// It is a foreground pixel
				++samples[yRow][xCol][NUM_SAMPLES];

				// Set background pixel to 255
				//m_mask.at<uchar>(i, j) = 255;
				m_mask.at<uchar>(yRow, xCol) = 255;

				//如果某个像素点连续N次被检测为前景，则认为一块静止区域被误判为运动，将其更新为背景点
				if (samples[yRow][xCol][NUM_SAMPLES] > 50)
				{
					int random = rng.uniform(0, SUBSAMPLE_FACTOR);
					if (random == 0)
					{
						random = rng.uniform(0, NUM_SAMPLES);
						//samples[i][j][random] = _image.at<uchar>(i, j);
						samples[yRow][xCol][random] = gray;

					}
				}
			}
		}
	}
}