#include "VideoProcessor.h"

int main()
{
	VideoProcessor processor;//��Ƶ�����࣬�ǳ���Ŀ��
	processor.Init("images/output.avi");//��ʼ����Ƶ�����࣬���ȫ��ͼ��ȡ����������ȡ��ʼ����ViBe�ĳ�ʼ��
	if (processor.SetInput("images/output.avi") != 0)//����һ����Ƶ
	{
		return -1;
	}
	
	processor.SetDelay(1);//������֮֡����ӳ�
	processor.DisplayInput("original image");//����ԭʼͼ�񴰿�
	processor.DisplayOutputFront("front image");//����ǰ��ͼ�񴰿�
	processor.DisplayOutputBack("back image");//���ñ���ͼ�񴰿�

	clock_t t3, t4;
	t3 = clock();
	processor.Run();//��ʼ��ѭ������
	t4 = clock();
	cout << "��ʱ�䣺" << (double)(t4 - t3) / CLOCKS_PER_SEC << std::endl;


	return 0;

}