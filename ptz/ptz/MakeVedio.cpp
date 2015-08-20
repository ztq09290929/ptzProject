//  读取文件夹下的图片生成视频文件  
//  Author：www.icvpr.com  
//  Blog：  http://blog.csdn.net/icvpr    

#include <iostream>  
#include <string>  
#include <io.h>  

#include <opencv2/opencv.hpp>  

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	// 图片集  
	string fileFolderPath = "C:\\Users\\louis\\Desktop\\PTZ\\PTZ\\zoomInZoomOut\\input";
	string fileExtension = "jpg";
	string fileFolder = fileFolderPath + "\\*." + fileExtension;

	// 输出视频  
	string outputVideoName = "output.avi";

	// openCV video writer  
	VideoWriter writer;

	int codec = 0;
	int frameRate = 25;
	Size frameSize;


	// 遍历文件夹  
	char fileName[1000];

	struct _finddata_t fileInfo;    // 文件信息结构体  

	// 1. 第一次查找  
	long findResult = _findfirst(fileFolder.c_str(), &fileInfo);
	if (findResult == -1)
	{
		_findclose(findResult);
		return -1;
	}

	// 2. 循环查找  
	do
	{
		sprintf(fileName, "%s\\%s", fileFolderPath.c_str(), fileInfo.name);

		if (fileInfo.attrib == _A_ARCH)  // 是存档类型文件  
		{
			Mat frame;
			frame = imread(fileName);    // 读入图片  
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

			// 将图片数据写入  
			writer.write(frame);

			// 显示  
			imshow("video", frame);
			waitKey(frameRate);
		}

	} while (!_findnext(findResult, &fileInfo));


	_findclose(findResult);


	return 0;
}