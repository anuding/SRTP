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
	QString path = QFileDialog::getOpenFileName(this, tr(u8"选择视频 "), ".", 
		tr("Video Files(*.mp4 *.png *.bmp *.jpg *.tif *.GIF )"));
	if (path.length() == 0) {	
	}
	else {
		ui.Edt_VideoAddress->setText(path);
	}
	vector<Mat> temp=concentration(path);//得到撞车帧
	//showImageList(temp);//显示撞车帧

	//QImage* img = new QImage;
	//img->load(path);
	//ui.Lab_VideoSummary->setPixmap(QPixmap::fromImage(*img));
	

		QListWidget *imageList = ui.Lst_KeyFrames;
	imageList->resize(365, 400);
	//设置QListWidget的显示模式  
	imageList->setViewMode(QListView::IconMode);
	//设置QListWidget中单元项的图片大小  
	imageList->setIconSize(QSize(100, 100));
	//设置QListWidget中单元项的间距  
	imageList->setSpacing(10);
	//设置自动适应布局调整（Adjust适应，Fixed不适应），默认不适应  
	imageList->setResizeMode(QListWidget::Adjust);
	//设置不能移动  
	imageList->setMovement(QListWidget::Static);
	for (auto tmp : temp)
	{
		QImage picQImage;
		QPixmap picQPixmap;

		cvtColor(tmp, tmp, CV_BGR2RGB);//三通道图片需bgr翻转成rgb
		picQImage = QImage((uchar*)tmp.data, tmp.cols, tmp.rows, QImage::Format_RGB888);
		picQPixmap = QPixmap::fromImage(picQImage);


		//定义QListWidgetItem对象  
		QListWidgetItem *imageItem = new QListWidgetItem;
		//ui.Lst_KeyFrames
		//为单元项设置属性  
		imageItem->setIcon(QIcon(picQPixmap));
		//imageItem->setText(tr("Browse"));  
		//重新设置单元项图片的宽度和高度  
		imageItem->setSizeHint(QSize(100, 120));
		//将单元项添加到QListWidget中  
		imageList->addItem(imageItem);
	}
}


void showImageList(vector<Mat> pic)
{
	//定义QListWidget对象  
	QListWidget *imageList = new QListWidget;
	imageList->resize(365, 400);
	//设置QListWidget的显示模式  
	imageList->setViewMode(QListView::IconMode);
	//设置QListWidget中单元项的图片大小  
	imageList->setIconSize(QSize(100, 100));
	//设置QListWidget中单元项的间距  
	imageList->setSpacing(10);
	//设置自动适应布局调整（Adjust适应，Fixed不适应），默认不适应  
	imageList->setResizeMode(QListWidget::Adjust);
	//设置不能移动  
	imageList->setMovement(QListWidget::Static);
	for (auto tmp : pic)
	{
		QImage picQImage;
		QPixmap picQPixmap;

		cvtColor(tmp, tmp, CV_BGR2RGB);//三通道图片需bgr翻转成rgb
		picQImage = QImage((uchar*)tmp.data, tmp.cols, tmp.rows, QImage::Format_RGB888);
		picQPixmap = QPixmap::fromImage(picQImage);


		//定义QListWidgetItem对象  
		QListWidgetItem *imageItem = new QListWidgetItem;
		//ui.Lst_KeyFrames
		//为单元项设置属性  
		imageItem->setIcon(QIcon(picQPixmap));
		//imageItem->setText(tr("Browse"));  
		//重新设置单元项图片的宽度和高度  
		imageItem->setSizeHint(QSize(100, 120));
		//将单元项添加到QListWidget中  
		imageList->addItem(imageItem);
	}
	//显示QListWidget  
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
	// 查找文件目录下的所有视频文件 
	//vector<string> videoPathStr = FindAllFile((videoPath + videoSuffix).c_str(), true);
	// 先读取一个视频文件，用于获取相关的参数 
	VideoCapture capture(string((const char *)Path_video.toLocal8Bit()));
	// 视频大小 
	Size videoSize(capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	



	Mat preFrame;
	bool stop(false);

	double totleFrameNum = capture.get(CV_CAP_PROP_FRAME_COUNT); // 获取视频总帧数
	int KeyframeNum = 0;
	vector<Mat> imgs(totleFrameNum);

	for (int frameNum = 0; frameNum < totleFrameNum; frameNum++)
	{
		Mat imgSrc;
		capture >> imgSrc; // 读一视频的一帧 
		if (!imgSrc.data)
			break;
		Mat frame;
		cvtColor(imgSrc, frame, CV_BGR2GRAY);

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



		if ((matArea / 230400) >= 17) // 面积比例大于10% 
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



