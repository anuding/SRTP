// compareHist.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <conio.h> 

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;


void compareHist();
int averagePic();

int main() 
{
	if (averagePic()==-1)//求解平均图
		return 0;

	Mat frame;
	
	CvCapture *capture = cvCreateFileCapture("E:/Code Project/SRTP/DEMO/DEMO/TEST00.mp4");
	double totalframes=cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_COUNT);// 获取总帧数
	
	cvNamedWindow("原图", 0);

	IplImage * Img = cvQueryFrame(capture);
	

	for (int i = 0; i <totalframes; i++)//统计背景模型
	{
		Img = cvQueryFrame(capture);
		int key = waitKey(24);
		if (key == 'q' || key == 'Q' || key == 27)
			break;
		
		if (!Img)
			break;
		cvShowImage("原图", Img);
		cvSaveImage("f1.jpg", Img);
		
		compareHist();

	}

	cvDestroyAllWindows();
	return 0;
}

void compareHist()
{
	Mat src_base, hsv_base;
	Mat src_test1, hsv_test1,src_test2;



	src_base = imread("model.jpg", 1);
	src_test1 = imread("f1.jpg", 1);

	/// 转换到 HSV
	cvtColor(src_base, hsv_base, CV_BGR2HSV);
	cvtColor(src_test1, hsv_test1, CV_BGR2HSV);


	/// 对hue通道使用30个bin,对saturatoin通道使用32个bin
	int h_bins = 50; int s_bins = 60;
	int histSize[] = { h_bins, s_bins };

	// hue的取值范围从0到256, saturation取值范围从0到180
	float h_ranges[] = { 0, 256 };
	float s_ranges[] = { 0, 180 };

	const float* ranges[] = { h_ranges, s_ranges };

	// 使用第0和第1通道
	int channels[] = { 0, 1 };

	/// 直方图
	MatND hist_base;
	MatND hist_test1;


	/// 计算HSV图像的直方图
	calcHist(&hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false);
	normalize(hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat());

	

	calcHist(&hsv_test1, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false);
	normalize(hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat());

	///应用不同的直方图对比方法
	for (int i = 0; i < 1; i++)
	{
		int compare_method = i;
		double base_base = compareHist(hist_base, hist_base, compare_method);
	
		double base_test1 = compareHist(hist_base, hist_test1, compare_method);
	
		printf(" Method [%d] Perfect, Base-Test(1): %f, %f\n", i, base_base, base_test1);

	}

	printf("Done \n");
	
}







//////////////////////////////

int averagePic()
{


	int nframe = 20;//利用前nfram帧求解平均图
	CvCapture *capture = cvCreateFileCapture("E:/Code Project/SRTP/DEMO/DEMO/TEST00.mp4");
	if (NULL == capture)
	{
		printf("没有找到该视频！\n");
		return -1;
	}
	IplImage * Img = cvQueryFrame(capture);

	IplImage * img_sum = cvCreateImage(cvGetSize(Img), IPL_DEPTH_32F, 3);
	cvZero(img_sum);
	for (int i = 0; i <nframe; i++)//统计背景模型
	{
		cvAcc(Img, img_sum);
		Img = cvQueryFrame(capture);
		cvWaitKey(10);
		
	}
	IplImage * img_sum_gray = cvCreateImage(cvGetSize(Img), IPL_DEPTH_8U, 3);
	cvConvertScale(img_sum, img_sum_gray, 1.0 / nframe);

	cvSaveImage("model.jpg", img_sum_gray);
	return 0;
}
