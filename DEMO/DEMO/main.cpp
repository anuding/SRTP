// compareHist.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <conio.h> 
#include <math.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;


void compareHist();
int averagePic();
void config();
void concentration();
void colorClustering();
void pointClustering();
void MultiImage_OneWin(const std::string& MultiShow_WinName, const vector<Mat>& SrcImg_V, CvSize SubPlot, CvSize ImgMax_Size = cvSize(400, 280));


int main() 
{

	cout<<CV_VERSION<<endl;
	//pointClustering();
	//colorClustering();
	concentration();


	//if (averagePic()==-1)//求解平均图
	//	return 0;

	//Mat frame;

	//CvCapture *capture = cvCreateFileCapture("TEST00.mp4");
	//double totalframes=cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_COUNT);// 获取总帧数
	//
	//cvNamedWindow("原图", 0);

	//IplImage * Img = cvQueryFrame(capture);
	//

	//for (int i = 0; i <totalframes; i++)//统计背景模型
	//{
	//	Img = cvQueryFrame(capture);
	//	int key = waitKey(24);
	//	if (key == 'q' || key == 'Q' || key == 27)
	//		break;
	//	
	//	if (!Img)
	//		break;
	//	cvShowImage("原图", Img);
	//	cvSaveImage("f1.jpg", Img);
	//	
	//	compareHist();

	//}

	//cvDestroyAllWindows();
	//return 0;
}

void pointClustering()
{
	const int MAX_CLUSTERS = 5;
	Scalar colorTab[] =     //因为最多只有5类，所以最多也就给5个颜色
	{
		Scalar(0, 0, 255),
		Scalar(0,255,0),
		Scalar(255,100,100),
		Scalar(255,0,255),
		Scalar(0,255,255)
	};

	Mat img(500, 500, CV_8UC3);
	RNG rng(12345); //随机数产生器

	for (;;)
	{
		int k, clusterCount = rng.uniform(2, MAX_CLUSTERS + 1);
		int i, sampleCount = rng.uniform(1, 1001);
		Mat points(sampleCount, 1, CV_32FC2), labels;   //产生的样本数，实际上为2通道的列向量，元素类型为Point2f

		clusterCount = MIN(clusterCount, sampleCount);
		Mat centers(clusterCount, 1, points.type());    //用来存储聚类后的中心点

		/* generate random sample from multigaussian distribution */
		for (k = 0; k < clusterCount; k++) //产生随机数
		{
			Point center;
			center.x = rng.uniform(0, img.cols);
			center.y = rng.uniform(0, img.rows);
			Mat pointChunk = points.rowRange(k*sampleCount / clusterCount,
				k == clusterCount - 1 ? sampleCount :
				(k + 1)*sampleCount / clusterCount);   //最后一个类的样本数不一定是平分的，
													   //剩下的一份都给最后一类
													   //每一类都是同样的方差，只是均值不同而已
			rng.fill(pointChunk, CV_RAND_NORMAL, Scalar(center.x, center.y), Scalar(img.cols*0.05, img.rows*0.05));
		}

		//randShuffle(points, 1, &rng);   //因为要聚类，所以先随机打乱points里面的点，注意points和pointChunk是共用数据的。

		kmeans(points, clusterCount, labels,
			TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),
			3, KMEANS_PP_CENTERS, centers);  //聚类3次，取结果最好的那次，聚类的初始化采用PP特定的随机算法。

		img = Scalar::all(0);

		for (i = 0; i < sampleCount; i++)
		{
			int clusterIdx = labels.at<int>(i);
			Point ipt = points.at<Point2f>(i);
			circle(img, ipt, 2, colorTab[clusterIdx], CV_FILLED, CV_AA);
		}

		imshow("clusters", img);

		char key = (char)waitKey();     //无限等待
		if (key == 27 || key == 'q' || key == 'Q') // 'ESC'
			break;
	}
}

void colorClustering()
{
	//VideoCapture capture("E:/Code_Project/SRTP/CODES/DEMO/DEMO/TEST100.mp4");
	Mat img = imread("f1.jpg", CV_LOAD_IMAGE_UNCHANGED);
	//capture >> img;
	Mat samples(img.cols*img.rows, 1, CV_32FC3);
	//标记矩阵，32位整形 
	Mat labels(img.cols*img.rows, 1, CV_32SC1);

	uchar* p;
	int i, j, k = 0;
	for (i = 0; i < img.rows; i++)
	{
		p = img.ptr<uchar>(i);
		for (j = 0; j< img.cols; j++)
		{
			samples.at<Vec3f>(k, 0)[0] = float(p[j * 3]);
			samples.at<Vec3f>(k, 0)[1] = float(p[j * 3 + 1]);
			samples.at<Vec3f>(k, 0)[2] = float(p[j * 3 + 2]);
			k++;
		}
	}
	Mat a();
	int clusterCount = 3;
	Mat centers(clusterCount, 1, samples.type());
	kmeans(samples, clusterCount, labels,
		TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),
		3, KMEANS_PP_CENTERS, centers);

//	最后我们把不同的簇用不同灰度来表示，并把结果放在img1中。

		//我们已知有3个聚类，用不同的灰度层表示。 
		Mat img1(img.rows, img.cols, CV_8UC1);
	float step = 255 / (clusterCount - 1);
	k = 0;
	for (i = 0; i < img1.rows; i++)
	{
		p = img1.ptr<uchar>(i);
		for (j = 0; j< img1.cols; j++)
		{
			int tt = labels.at<int>(k, 0);
			k++;
			p[j] = 255 - tt*step;
		}
	}

	namedWindow("image1");
	imshow("image1", img1);
	waitKey();
}

void config()
{//E:/Code_Project/SRTP/CODES/DEMO/DEMO/
	FileStorage fs("config.xml", FileStorage::READ);
	bool flag = fs.isOpened();
	cout << "flag = " << flag << endl << endl;
	// if failed to open a file  
	if (!fs.isOpened()) {
		cout << "failed to open file test.yml " << endl << endl;
		return ;
	}
	string videoPath;
	string videoSuffix;
	Rect roiRect;
	string imgSavePath;

	fs["videoReadPath"] >> videoPath;
	fs["videoSuffix"] >> videoSuffix;
	fs["imgSavePath"] >> imgSavePath;
	fs["roi"] >> roiRect;

	cout << videoPath << endl
		<< videoSuffix << endl
		<< imgSavePath << endl
		<< roiRect << endl;
}

void concentration()
{
	FileStorage fs("config.xml", FileStorage::READ);
	bool flag = fs.isOpened();
	cout << "flag = " << flag << endl << endl;
	// if failed to open a file  
	if (!fs.isOpened()) {
		cout << "failed to open file test.yml " << endl << endl;
		return;
	}
	string videoPath;
	string videoSuffix;
	Rect roiRect;
	string imgSavePath;

	fs["videoReadPath"] >> videoPath;
	fs["videoSuffix"] >> videoSuffix;
	fs["imgSavePath"] >> imgSavePath;
	fs["roi"] >> roiRect;
	// 查找文件目录下的所有视频文件 
	//vector<string> videoPathStr = FindAllFile((videoPath + videoSuffix).c_str(), true);
	// 先读取一个视频文件，用于获取相关的参数 
	VideoCapture capture("E:/Code_Project/SRTP/CODES/DEMO/DEMO/TEST200.mp4");
	// 视频大小 
	Size videoSize(capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	// 创建一个视频写入对象 
	VideoWriter writer("../result.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, videoSize);

	//for (auto videoName : "E:/Code_Project/SRTP/CODES/DEMO/DEMO/TEST00.mp4")
	{
		//capture.open(videoName); // 读入路径下的视频
		


		
		Mat preFrame;
		bool stop(false);

		double totleFrameNum = capture.get(CV_CAP_PROP_FRAME_COUNT); // 获取视频总帧数
		int coun = 0,q;
		vector<Mat> imgs(50);

		for (int frameNum = 0; frameNum < totleFrameNum; frameNum++)
		{
			Mat imgSrc;
			capture >> imgSrc; // 读一视频的一帧 
			if (!imgSrc.data)
				break;
			Mat frame;
			cvtColor(imgSrc, frame, CV_BGR2GRAY);
			//++frameNum;
			if (frameNum == 0)
			{
				preFrame = frame;
			}
			Mat frameDif;
			absdiff(frame, preFrame, frameDif); // 帧差法 
			preFrame = frame;

			threshold(frameDif, frameDif, 30, 255, THRESH_BINARY); 

			Mat imgRoi = frameDif(roiRect);
			double matArea = cv::sum(imgRoi)[0]; 
			
				
			
			if ((matArea / 230400) >= 15) // 面积比例大于10% 
			{
				
				imgs[coun] = imgSrc;
				coun++;
				//imshow(name, imgSrc);
				//waitKey(0);

				//销毁MyWindow的窗口

				//destroyWindow("MyWindow");
				
				cout << "frameNum= "<< frameNum <<"   "<<(matArea / 230400) << endl;
			}
			q = frameNum;
		}
		vector<Mat> timgs(coun);
		for (int i = 0; i < coun; i++)
		{
			timgs[i] = imgs[i];
		}
		int row = sqrt(coun)+1;
		MultiImage_OneWin("Multiple Images", timgs, cvSize(row, row), cvSize(200,200));
		cout << endl << endl << "q=" << q << endl;
		coun = 0;
	}
	
	capture.release();
	writer.release();
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
	CvCapture *capture = cvCreateFileCapture("E:/Code_Project/SRTP/CODES/DEMO/DEMO/TEST00.mp4");
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



void MultiImage_OneWin(const std::string& MultiShow_WinName, const vector<Mat>& SrcImg_V, CvSize SubPlot, CvSize ImgMax_Size)
{
	//Reference : http://blog.csdn.net/yangyangyang20092010/article/details/21740373  

	//************* Usage *************//  
	//vector<Mat> imgs(4);  
	//imgs[0] = imread("F:\\SA2014.jpg");  
	//imgs[1] = imread("F:\\SA2014.jpg");  
	//imgs[2] = imread("F:\\SA2014.jpg");  
	//imgs[3] = imread("F:\\SA2014.jpg");  
	//MultiImage_OneWin("T", imgs, cvSize(2, 2), cvSize(400, 280));  

	//Window's image  
	Mat Disp_Img;
	//Width of source image  
	CvSize Img_OrigSize = cvSize(SrcImg_V[0].cols, SrcImg_V[0].rows);
	//******************** Set the width for displayed image ********************//  
	//Width vs height ratio of source image  
	float WH_Ratio_Orig = Img_OrigSize.width / (float)Img_OrigSize.height;
	CvSize ImgDisp_Size = cvSize(100, 100);
	if (Img_OrigSize.width > ImgMax_Size.width)
		ImgDisp_Size = cvSize(ImgMax_Size.width, (int)ImgMax_Size.width / WH_Ratio_Orig);
	else if (Img_OrigSize.height > ImgMax_Size.height)
		ImgDisp_Size = cvSize((int)ImgMax_Size.height*WH_Ratio_Orig, ImgMax_Size.height);
	else
		ImgDisp_Size = cvSize(Img_OrigSize.width, Img_OrigSize.height);
	//******************** Check Image numbers with Subplot layout ********************//  
	int Img_Num = (int)SrcImg_V.size();
	if (Img_Num > SubPlot.width * SubPlot.height)
	{
		cout << "Your SubPlot Setting is too small !" << endl;
		exit(0);
	}
	//******************** Blank setting ********************//  
	CvSize DispBlank_Edge = cvSize(80, 60);
	CvSize DispBlank_Gap = cvSize(15, 15);
	//******************** Size for Window ********************//  
	Disp_Img.create(Size(ImgDisp_Size.width*SubPlot.width + DispBlank_Edge.width + (SubPlot.width - 1)*DispBlank_Gap.width,
		ImgDisp_Size.height*SubPlot.height + DispBlank_Edge.height + (SubPlot.height - 1)*DispBlank_Gap.height), CV_8UC3);
	Disp_Img.setTo(0);//Background  
					  //Left top position for each image  
	int EdgeBlank_X = (Disp_Img.cols - (ImgDisp_Size.width*SubPlot.width + (SubPlot.width - 1)*DispBlank_Gap.width)) / 2;
	int EdgeBlank_Y = (Disp_Img.rows - (ImgDisp_Size.height*SubPlot.height + (SubPlot.height - 1)*DispBlank_Gap.height)) / 2;
	CvPoint LT_BasePos = cvPoint(EdgeBlank_X, EdgeBlank_Y);
	CvPoint LT_Pos = LT_BasePos;

	//Display all images  
	for (int i = 0; i < Img_Num; i++)
	{
		//Obtain the left top position  
		if ((i%SubPlot.width == 0) && (LT_Pos.x != LT_BasePos.x))
		{
			LT_Pos.x = LT_BasePos.x;
			LT_Pos.y += (DispBlank_Gap.height + ImgDisp_Size.height);
		}
		//Writting each to Window's Image  
		Mat imgROI = Disp_Img(Rect(LT_Pos.x, LT_Pos.y, ImgDisp_Size.width, ImgDisp_Size.height));
		resize(SrcImg_V[i], imgROI, Size(ImgDisp_Size.width, ImgDisp_Size.height));

		LT_Pos.x += (DispBlank_Gap.width + ImgDisp_Size.width);
	}

	//Get the screen size of computer  
	int Scree_W = 1920;// GetSystemMetrics(SM_CXSCREEN);
	int Scree_H = 1080;// GetSystemMetrics(SM_CYSCREEN);

	cvNamedWindow(MultiShow_WinName.c_str(), CV_WINDOW_AUTOSIZE);
	cvMoveWindow(MultiShow_WinName.c_str(), (Scree_W - Disp_Img.cols) / 2, (Scree_H - Disp_Img.rows) / 2);//Centralize the window  
	cvShowImage(MultiShow_WinName.c_str(), &(IplImage(Disp_Img)));
	cvWaitKey(0);
	cvDestroyWindow(MultiShow_WinName.c_str());
}