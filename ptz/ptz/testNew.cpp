#include "VideoProcessor.h"

int main()
{
	VideoProcessor processor;
	processor.Init("images/output2.avi");
	if (processor.SetInput("images/output2.avi") != 0)
	{
		return -1;
	}
	
	processor.SetDelay(1);
	processor.DisplayInput("original image");
	processor.DisplayOutputFront("front image");
	processor.DisplayOutputBack("back image");

	clock_t t3, t4;
	t3 = clock();
	processor.Run();
	t4 = clock();
	cout << "×ÜÊ±¼ä£º" << (double)(t4 - t3) / CLOCKS_PER_SEC << std::endl;


	return 0;

}