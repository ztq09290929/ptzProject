#include "ViBe2.h"

using namespace std;
using namespace cv;

int c_xoff[9] = { -1, 0, 1, -1, 1, -1, 0, 1, 0 };  //x���ھӵ�
int c_yoff[9] = { -1, 0, 1, -1, 1, -1, 0, 1, 0 };  //y���ھӵ�

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
void ViBe_BGS::init(const Mat _image, Mat frame)
{
	//���ȼ�¼���䶯̬�����������, ����ά��NUM_SAMPLES + 1��������ʱ�����ͷ��ڴ�
	imgRows = _image.rows;
	imgCols = _image.cols;
	//��̬������ά���飬samples[][][NUM_SAMPLES]�洢ǰ�����������Ĵ���
	samples = new unsigned char **[_image.rows];
	for (int i = 0; i < _image.rows; i++)
	{
		samples[i] = new unsigned char *[_image.cols];
		for (int j = 0; j < _image.cols; j++)
		{
			samples[i][j] = new unsigned char[NUM_SAMPLES + 1];//���һ���洢ǰ��������
			for (int k = 0; k < NUM_SAMPLES + 1; k++)
			{
				samples[i][j][k] = 0;
			}

		}

	}
	m_mask = Mat::zeros(_image.size(), CV_8UC1);
	m_fore = Mat::zeros(frame.size(), CV_8UC1);
	return;
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
		int xCol = (int)(_image[i].x + 0.5);//�õ���һ����������֮���x��y�����Լ��Ҷ�ֵ
		int yRow = (int)(_image[i].y + 0.5);
		uchar gray = (uchar)_image[i].z;
		if (0 <= xCol && xCol < imgCols && 0 <= yRow && yRow < imgRows)//����õ��ڱ���֮�ڣ���������
		{
			int matches(0), count(0);
			int dist;

			while (matches < MIN_MATCHES && count < NUM_SAMPLES)
			{
				//dist = abs(samples[i][j][count] - _image.at<uchar>(i, j));
				dist = abs(samples[yRow][xCol][count] - gray);//����ָ�������������
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
				m_fore.at<uchar>((int)i / m_fore.cols, (int)i%m_fore.cols) = 0;
				// ���һ�������Ǳ����㣬��ô���� 1 / defaultSubsamplingFactor �ĸ���ȥ�����Լ���ģ������ֵ
				int random = rng.uniform(0, SUBSAMPLE_FACTOR);
				if (random == 0)
				{
					random = rng.uniform(0, NUM_SAMPLES);
					//samples[i][j][random] = _image.at<uchar>(i, j);
					samples[yRow][xCol][random] = gray;//ʹ��ָ���������

				}

				// ͬʱҲ�� 1 / defaultSubsamplingFactor �ĸ���ȥ���������ھӵ��ģ������ֵ
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
				m_fore.at<uchar>((int)i / m_fore.cols, (int)i%m_fore.cols) = 255;
				//���ĳ�����ص�����N�α����Ϊǰ��������Ϊһ�龲ֹ��������Ϊ�˶����������Ϊ������
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