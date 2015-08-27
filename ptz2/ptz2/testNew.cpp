#include "VideoProcessor.h"

int main()
{
	VideoProcessor processor;//视频播放类，是程序的框架
	processor.Init("images/outputno.avi");//初始化视频播放类，完成全景图提取，特征点提取初始化和ViBe的初始化
	if (processor.SetInput("images/outputno.avi") != 0)//读入一个视频
	{
		return -1;
	}
	
	processor.SetDelay(1);//设置两帧之间的延迟
	processor.DisplayInput("original image");//设置原始图像窗口
	processor.DisplayOutputFront("front image");//设置前景图像窗口
	processor.DisplayOutputBack("back image");//设置背景图像窗口

	clock_t t3, t4;
	t3 = clock();
	processor.Run();//开始大循环处理
	t4 = clock();
	cout << "总时间：" << (double)(t4 - t3) / CLOCKS_PER_SEC << std::endl;


	return 0;

}