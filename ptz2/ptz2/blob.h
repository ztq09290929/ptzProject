#pragma once
#include "opencv2/video/background_segm.hpp"
#include "opencv2/legacy/blobtrack.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc_c.h>

class CBlob
{
public:
	void BlobDetecter(const cv::Mat& _binImg, cv::Mat& _outputImg);
};



