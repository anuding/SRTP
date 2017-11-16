#include "Image.h"

void main()
{
	Image img("E:/Code Project/SRTP/ImageFeature-master/img.jpg");
	img.showImage();
	img.showGrayHist();//灰度直方
	//img.showRGBGrayHist();//颜色直方
	//img.showHSHist();//HSV直方
	//img.showColorMoment();//颜色矩
	//img.showCCV();//聚合向量CCV
	//img.showColorCorrelogram();//颜色相关
	system("pause");
}