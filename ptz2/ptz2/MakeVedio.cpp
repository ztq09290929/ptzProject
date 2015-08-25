//  ��ȡ�ļ����µ�ͼƬ������Ƶ�ļ�  
//  Author��www.icvpr.com  
//  Blog��  http://blog.csdn.net/icvpr    

#include <iostream>  
#include <string>  
#include <io.h>  

#include <opencv2/opencv.hpp>  

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	// ͼƬ��  
	string fileFolderPath = "C:\\Users\\louis\\Desktop\\PTZ\\PTZ\\zoomInZoomOut\\input";
	string fileExtension = "jpg";
	string fileFolder = fileFolderPath + "\\*." + fileExtension;

	// �����Ƶ  
	string outputVideoName = "output.avi";

	// openCV video writer  
	VideoWriter writer;

	int codec = 0;
	int frameRate = 25;
	Size frameSize;


	// �����ļ���  
	char fileName[1000];

	struct _finddata_t fileInfo;    // �ļ���Ϣ�ṹ��  

	// 1. ��һ�β���  
	long findResult = _findfirst(fileFolder.c_str(), &fileInfo);
	if (findResult == -1)
	{
		_findclose(findResult);
		return -1;
	}

	// 2. ѭ������  
	do
	{
		sprintf(fileName, "%s\\%s", fileFolderPath.c_str(), fileInfo.name);

		if (fileInfo.attrib == _A_ARCH)  // �Ǵ浵�����ļ�  
		{
			Mat frame;
			frame = imread(fileName);    // ����ͼƬ  
			if (!writer.isOpened())
			{
				frameSize.width = frame.cols;
				frameSize.height = frame.rows;

				if (!writer.open(outputVideoName, CV_FOURCC('D', 'I', 'V', 'X'), frameRate, frameSize, true))
				{
					cout << "open writer error..." << endl;
					return -1;
				}
			}

			// ��ͼƬ����д��  
			writer.write(frame);

			// ��ʾ  
			imshow("video", frame);
			waitKey(frameRate);
		}

	} while (!_findnext(findResult, &fileInfo));


	_findclose(findResult);


	return 0;
}