#ifndef __IMAGE_H__
#define __IMAGE_H__ 1
//#include <opencv2/opencv.hpp>  
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <queue>
#define Num_ColorBin 32 //* quantized to 32 colors */
#define Distance_Range 10 //* distance range to calculate in correlogram *//
using namespace std;

class Image
{
public:
	Image();
	Image(char* file);
	char* getImagePath();				// ��ȡͼƬ�ļ�·��
	void setImagePath(char* file);		// ����ͼƬ�ļ�·��
	void showImage();					// ��ʾͼƬ
	void showGrayHist();				// ��ʾͼƬ�Ҷ�ֱ��ͼ
	void showRGBGrayHist();				// ��ʾ��ɫֱ��ͼ
	void showHSHist();					// ��ʾHSֱ��ͼ
	void showColorMoment();				// ��ʾ��ɫ��
	// ��ɫ����,<<����ɫ��������������ͼ�����>>,�����£���С��������毣����
	void colorQuantization(IplImage** planes, IplImage* quantized);
	void showCCV();						// ��ʾ��ɫ�ۺ�����
	void showColorCorrelogram();		// ��ʾ��ɫ���ͼ
	~Image();

private:
	char* image;						// ͼƬ�ļ�·��
};

#endif