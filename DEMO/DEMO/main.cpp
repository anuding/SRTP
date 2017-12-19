// compareHist.cpp : �������̨Ӧ�ó������ڵ㡣
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
void config();
void concentration();
void colorClustering();
void pointClustering();



int main() 
{

	cout<<CV_VERSION<<endl;
	pointClustering();
	//colorClustering();
	//concentration();


	//if (averagePic()==-1)//���ƽ��ͼ
	//	return 0;

	//Mat frame;

	//CvCapture *capture = cvCreateFileCapture("TEST00.mp4");
	//double totalframes=cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_COUNT);// ��ȡ��֡��
	//
	//cvNamedWindow("ԭͼ", 0);

	//IplImage * Img = cvQueryFrame(capture);
	//

	//for (int i = 0; i <totalframes; i++)//ͳ�Ʊ���ģ��
	//{
	//	Img = cvQueryFrame(capture);
	//	int key = waitKey(24);
	//	if (key == 'q' || key == 'Q' || key == 27)
	//		break;
	//	
	//	if (!Img)
	//		break;
	//	cvShowImage("ԭͼ", Img);
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
	Scalar colorTab[] =     //��Ϊ���ֻ��5�࣬�������Ҳ�͸�5����ɫ
	{
		Scalar(0, 0, 255),
		Scalar(0,255,0),
		Scalar(255,100,100),
		Scalar(255,0,255),
		Scalar(0,255,255)
	};

	Mat img(500, 500, CV_8UC3);
	RNG rng(12345); //�����������

	for (;;)
	{
		int k, clusterCount = rng.uniform(2, MAX_CLUSTERS + 1);
		int i, sampleCount = rng.uniform(1, 1001);
		Mat points(sampleCount, 1, CV_32FC2), labels;   //��������������ʵ����Ϊ2ͨ������������Ԫ������ΪPoint2f

		clusterCount = MIN(clusterCount, sampleCount);
		Mat centers(clusterCount, 1, points.type());    //�����洢���������ĵ�

		/* generate random sample from multigaussian distribution */
		for (k = 0; k < clusterCount; k++) //���������
		{
			Point center;
			center.x = rng.uniform(0, img.cols);
			center.y = rng.uniform(0, img.rows);
			Mat pointChunk = points.rowRange(k*sampleCount / clusterCount,
				k == clusterCount - 1 ? sampleCount :
				(k + 1)*sampleCount / clusterCount);   //���һ�������������һ����ƽ�ֵģ�
													   //ʣ�µ�һ�ݶ������һ��
													   //ÿһ�඼��ͬ���ķ��ֻ�Ǿ�ֵ��ͬ����
			rng.fill(pointChunk, CV_RAND_NORMAL, Scalar(center.x, center.y), Scalar(img.cols*0.05, img.rows*0.05));
		}

		//randShuffle(points, 1, &rng);   //��ΪҪ���࣬�������������points����ĵ㣬ע��points��pointChunk�ǹ������ݵġ�

		kmeans(points, clusterCount, labels,
			TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),
			3, KMEANS_PP_CENTERS, centers);  //����3�Σ�ȡ�����õ��ǴΣ�����ĳ�ʼ������PP�ض�������㷨��

		img = Scalar::all(0);

		for (i = 0; i < sampleCount; i++)
		{
			int clusterIdx = labels.at<int>(i);
			Point ipt = points.at<Point2f>(i);
			circle(img, ipt, 2, colorTab[clusterIdx], CV_FILLED, CV_AA);
		}

		imshow("clusters", img);

		char key = (char)waitKey();     //���޵ȴ�
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
	//��Ǿ���32λ���� 
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

//	������ǰѲ�ͬ�Ĵ��ò�ͬ�Ҷ�����ʾ�����ѽ������img1�С�

		//������֪��3�����࣬�ò�ͬ�ĻҶȲ��ʾ�� 
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
	// �����ļ�Ŀ¼�µ�������Ƶ�ļ� 
	//vector<string> videoPathStr = FindAllFile((videoPath + videoSuffix).c_str(), true);
	// �ȶ�ȡһ����Ƶ�ļ������ڻ�ȡ��صĲ��� 
	VideoCapture capture("E:/Code_Project/SRTP/CODES/DEMO/DEMO/TEST100.mp4");
	// ��Ƶ��С 
	Size videoSize(capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	// ����һ����Ƶд����� 
	VideoWriter writer("../result.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, videoSize);

	//for (auto videoName : "E:/Code_Project/SRTP/CODES/DEMO/DEMO/TEST00.mp4")
	{
		//capture.open(videoName); // ����·���µ���Ƶ

		Mat preFrame;
		bool stop(false);

		double totleFrameNum = capture.get(CV_CAP_PROP_FRAME_COUNT); // ��ȡ��Ƶ��֡��
		int coun = 0,q;
		
		for (int frameNum = 0; frameNum < totleFrameNum; frameNum++)
		{
			Mat imgSrc;
			capture >> imgSrc; // ��һ��Ƶ��һ֡ 
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
			absdiff(frame, preFrame, frameDif); // ֡� 
			preFrame = frame;

			threshold(frameDif, frameDif, 30, 255, THRESH_BINARY); 

			Mat imgRoi = frameDif(roiRect);
			double matArea = cv::sum(imgRoi)[0]; 
			
				
			
			if ((matArea / 230400) >= 10) // �����������10% 
			{
				string name = "my";
				coun++;
				string num;
				stringstream stream;
				stream << coun;
				num = stream.str();   

				name = name + num + ".jpg";
				imwrite(name, imgSrc);
				
				cout << "frameNum= "<< frameNum <<"   "<<(matArea / 230400) << endl;
			}
			q = frameNum;
		}
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

	/// ת���� HSV
	cvtColor(src_base, hsv_base, CV_BGR2HSV);
	cvtColor(src_test1, hsv_test1, CV_BGR2HSV);


	/// ��hueͨ��ʹ��30��bin,��saturatoinͨ��ʹ��32��bin
	int h_bins = 50; int s_bins = 60;
	int histSize[] = { h_bins, s_bins };

	// hue��ȡֵ��Χ��0��256, saturationȡֵ��Χ��0��180
	float h_ranges[] = { 0, 256 };
	float s_ranges[] = { 0, 180 };

	const float* ranges[] = { h_ranges, s_ranges };

	// ʹ�õ�0�͵�1ͨ��
	int channels[] = { 0, 1 };

	/// ֱ��ͼ
	MatND hist_base;
	MatND hist_test1;


	/// ����HSVͼ���ֱ��ͼ
	calcHist(&hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false);
	normalize(hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat());

	

	calcHist(&hsv_test1, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false);
	normalize(hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat());

	///Ӧ�ò�ͬ��ֱ��ͼ�Աȷ���
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


	int nframe = 20;//����ǰnfram֡���ƽ��ͼ
	CvCapture *capture = cvCreateFileCapture("E:/Code_Project/SRTP/CODES/DEMO/DEMO/TEST00.mp4");
	if (NULL == capture)
	{
		printf("û���ҵ�����Ƶ��\n");
		return -1;
	}
	IplImage * Img = cvQueryFrame(capture);

	IplImage * img_sum = cvCreateImage(cvGetSize(Img), IPL_DEPTH_32F, 3);
	cvZero(img_sum);
	for (int i = 0; i <nframe; i++)//ͳ�Ʊ���ģ��
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
