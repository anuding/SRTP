#include "Image.h"

void main()
{
	Image img("E:/Code Project/SRTP/ImageFeature-master/img.jpg");
	img.showImage();
	img.showGrayHist();//�Ҷ�ֱ��
	//img.showRGBGrayHist();//��ɫֱ��
	//img.showHSHist();//HSVֱ��
	//img.showColorMoment();//��ɫ��
	//img.showCCV();//�ۺ�����CCV
	//img.showColorCorrelogram();//��ɫ���
	system("pause");
}