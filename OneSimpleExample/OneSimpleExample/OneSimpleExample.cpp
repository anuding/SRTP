// OneSimpleExample.cpp : �������̨Ӧ�ó������ڵ㡣
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
	// �����ļ�Ŀ¼�µ�������Ƶ�ļ� 
	//vector<string> videoPathStr = FindAllFile((videoPath + videoSuffix).c_str(), true);
	// �ȶ�ȡһ����Ƶ�ļ������ڻ�ȡ��صĲ��� 
	VideoCapture capture("E:/TEST00.mp4");

	// ��Ƶ��С 
	Size videoSize(capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	// ����һ����Ƶд����� 
	VideoWriter writer("../result.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, videoSize);

	//for (auto videoName : videoPathStr)
	{
		//capture.open(videoName); // ����·���µ���Ƶ

		Mat preFrame;
		bool stop(false);

		double totleFrameNum = capture.get(CV_CAP_PROP_FRAME_COUNT); // ��ȡ��Ƶ��֡��

		for (int frameNum = 0; frameNum < totleFrameNum; frameNum++)
		{
			Mat imgSrc;
			capture >> imgSrc; // ��һ��Ƶ��һ֡ 
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
			absdiff(frame, preFrame, frameDif); // ֡� 
			preFrame = frame;

			threshold(frameDif, frameDif, 30, 255, THRESH_BINARY); // ��ֵ��

			Mat imgRoi = frameDif(roiRect);
			//double matArea = computeMatArea(imgRoi); // �����������
			double matArea = cv::sum(imgRoi)[0] ; // �����������
			//mat�� area = cv::sum(mat)[0];
			if (matArea / (imgRoi.rows*imgRoi.cols) > 0.1) // �����������10% 
			{
				writer << frameDif;// д����Ƶ 
			}
		}
	}
	capture.release();
	writer.release();
}

