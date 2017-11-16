// OneSimpleExample.cpp : 定义控制台应用程序的入口点。
//
#include "stdafx.h"
#include <conio.h> 
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
using namespace std;
using namespace cv;

int main()
{
	FileStorage fs("../config.xml", FileStorage::WRITE);

	string videoPath;
	string videoSuffix;
	Rect roiRect;
	string imgSavePath;

	fs["videoReadPath"] >> videoPath;
	fs["videoSuffix"] >> videoSuffix;
	fs["imgSavePath"] >> imgSavePath;
	fs["roi"] >> roiRect;



	///////////////////////////////////////////////////////////////////////////////
	// 查找文件目录下的所有视频文件 
	//vector<string> videoPathStr = FindAllFile((videoPath + videoSuffix).c_str(), true);
	// 先读取一个视频文件，用于获取相关的参数 
	VideoCapture capture("E:/TEST00.mp4");

	// 视频大小 
	Size videoSize(capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	// 创建一个视频写入对象 
	VideoWriter writer("../result.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, videoSize);

	//for (auto videoName : videoPathStr)
	{
		//capture.open(videoName); // 读入路径下的视频

		Mat preFrame;
		bool stop(false);

		double totleFrameNum = capture.get(CV_CAP_PROP_FRAME_COUNT); // 获取视频总帧数

		for (int frameNum = 0; frameNum < totleFrameNum; frameNum++)
		{
			Mat imgSrc;
			capture >> imgSrc; // 读一视频的一帧 
			if (!imgSrc.data)
				break;
			Mat frame;
			cvtColor(imgSrc, frame, CV_BGR2GRAY);
			++frameNum;
			if (frameNum == 1)
			{
				preFrame = frame;
			}
			Mat frameDif;
			absdiff(frame, preFrame, frameDif); // 帧差法 
			preFrame = frame;

			threshold(frameDif, frameDif, 30, 255, THRESH_BINARY); // 二值化

			Mat imgRoi = frameDif(roiRect);
			//double matArea = computeMatArea(imgRoi); // 计算区域面积
			double matArea = cv::sum(imgRoi)[0] ; // 计算区域面积
			//mat， area = cv::sum(mat)[0];
			if (matArea / (imgRoi.rows*imgRoi.cols) > 0.1) // 面积比例大于10% 
			{
				writer << frameDif;// 写入视频 
			}
		}
	}
	capture.release();
	writer.release();
}

