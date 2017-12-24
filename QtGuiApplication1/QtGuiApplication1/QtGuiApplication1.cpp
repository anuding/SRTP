#include "QtGuiApplication1.h"
#include <iostream>
#include <qfiledialog.h>
#include "QMenu"  
#include "QMenuBar"  
#include "QAction"  
#include "QMessageBox"  
#include "QFileDialog"  
#include "QDebug"  
#include "QListWidget" 
#include <QApplication>  
#include <QProcess>  
#include <QMessageBox>  
#include <qmessagebox.h>
#include <math.h>
#include <conio.h> 

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
using namespace std;
using namespace cv;


vector<Mat> concentration(QString Path_video);
void showImageList(vector<Mat> pic);



QtGuiApplication1::QtGuiApplication1(QWidget *parent)
	: QMainWindow(parent)
{

	ui.setupUi(this);
}

void QtGuiApplication1::myExitButtonFuc()
{
	QString path = QFileDialog::getOpenFileName(this, tr(u8"ѡ����Ƶ "), ".", 
		tr("Video Files(*.mp4 *.png *.bmp *.jpg *.tif *.GIF )"));
	if (path.length() == 0) {	
	}
	else {
		ui.Edt_VideoAddress->setText(path);
	}
	vector<Mat> temp=concentration(path);//�õ�ײ��֡
	//showImageList(temp);//��ʾײ��֡

	//QImage* img = new QImage;
	//img->load(path);
	//ui.Lab_VideoSummary->setPixmap(QPixmap::fromImage(*img));
	

		QListWidget *imageList = ui.Lst_KeyFrames;
	imageList->resize(365, 400);
	//����QListWidget����ʾģʽ  
	imageList->setViewMode(QListView::IconMode);
	//����QListWidget�е�Ԫ���ͼƬ��С  
	imageList->setIconSize(QSize(100, 100));
	//����QListWidget�е�Ԫ��ļ��  
	imageList->setSpacing(10);
	//�����Զ���Ӧ���ֵ�����Adjust��Ӧ��Fixed����Ӧ����Ĭ�ϲ���Ӧ  
	imageList->setResizeMode(QListWidget::Adjust);
	//���ò����ƶ�  
	imageList->setMovement(QListWidget::Static);
	for (auto tmp : temp)
	{
		QImage picQImage;
		QPixmap picQPixmap;

		cvtColor(tmp, tmp, CV_BGR2RGB);//��ͨ��ͼƬ��bgr��ת��rgb
		picQImage = QImage((uchar*)tmp.data, tmp.cols, tmp.rows, QImage::Format_RGB888);
		picQPixmap = QPixmap::fromImage(picQImage);


		//����QListWidgetItem����  
		QListWidgetItem *imageItem = new QListWidgetItem;
		//ui.Lst_KeyFrames
		//Ϊ��Ԫ����������  
		imageItem->setIcon(QIcon(picQPixmap));
		//imageItem->setText(tr("Browse"));  
		//�������õ�Ԫ��ͼƬ�Ŀ�Ⱥ͸߶�  
		imageItem->setSizeHint(QSize(100, 120));
		//����Ԫ����ӵ�QListWidget��  
		imageList->addItem(imageItem);
	}
}


void showImageList(vector<Mat> pic)
{
	//����QListWidget����  
	QListWidget *imageList = new QListWidget;
	imageList->resize(365, 400);
	//����QListWidget����ʾģʽ  
	imageList->setViewMode(QListView::IconMode);
	//����QListWidget�е�Ԫ���ͼƬ��С  
	imageList->setIconSize(QSize(100, 100));
	//����QListWidget�е�Ԫ��ļ��  
	imageList->setSpacing(10);
	//�����Զ���Ӧ���ֵ�����Adjust��Ӧ��Fixed����Ӧ����Ĭ�ϲ���Ӧ  
	imageList->setResizeMode(QListWidget::Adjust);
	//���ò����ƶ�  
	imageList->setMovement(QListWidget::Static);
	for (auto tmp : pic)
	{
		QImage picQImage;
		QPixmap picQPixmap;

		cvtColor(tmp, tmp, CV_BGR2RGB);//��ͨ��ͼƬ��bgr��ת��rgb
		picQImage = QImage((uchar*)tmp.data, tmp.cols, tmp.rows, QImage::Format_RGB888);
		picQPixmap = QPixmap::fromImage(picQImage);


		//����QListWidgetItem����  
		QListWidgetItem *imageItem = new QListWidgetItem;
		//ui.Lst_KeyFrames
		//Ϊ��Ԫ����������  
		imageItem->setIcon(QIcon(picQPixmap));
		//imageItem->setText(tr("Browse"));  
		//�������õ�Ԫ��ͼƬ�Ŀ�Ⱥ͸߶�  
		imageItem->setSizeHint(QSize(100, 120));
		//����Ԫ����ӵ�QListWidget��  
		imageList->addItem(imageItem);
	}
	//��ʾQListWidget  
	imageList->show();
}



vector<Mat> concentration(QString Path_video)
{
	FileStorage fs("config.xml", FileStorage::READ);
	bool flag = fs.isOpened();
	Mat wrong;
	cout << "flag = " << flag << endl << endl;
	// if failed to open a file  
	if (!fs.isOpened()) {
		cout << "failed to open file test.yml " << endl << endl;
		return wrong;
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
	VideoCapture capture(string((const char *)Path_video.toLocal8Bit()));
	// ��Ƶ��С 
	Size videoSize(capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	



	Mat preFrame;
	bool stop(false);

	double totleFrameNum = capture.get(CV_CAP_PROP_FRAME_COUNT); // ��ȡ��Ƶ��֡��
	int KeyframeNum = 0;
	vector<Mat> imgs(totleFrameNum);

	for (int frameNum = 0; frameNum < totleFrameNum; frameNum++)
	{
		Mat imgSrc;
		capture >> imgSrc; // ��һ��Ƶ��һ֡ 
		if (!imgSrc.data)
			break;
		Mat frame;
		cvtColor(imgSrc, frame, CV_BGR2GRAY);

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



		if ((matArea / 230400) >= 17) // �����������10% 
		{
			imgs[KeyframeNum] = imgSrc;
			KeyframeNum++;
			cout << "frameNum= " << frameNum << "   " << (matArea / 230400) << endl;
		}
		//KeyframeNum = frameNum;
	}

	vector<Mat> imgsToShow(KeyframeNum);
	for (int i = 0; i < KeyframeNum; i++)
	{
		imgsToShow[i] = imgs[i];
	}
	int row = sqrt(KeyframeNum) + 1;

	//MultiImage_OneWin("Multiple Images", imgsToShow, cvSize(row, row), cvSize(200, 200));
	cout << endl << endl << "KeyframeNum=" << KeyframeNum << endl;
	//coun = 0;


	capture.release();
	return imgsToShow;

}



