//  局部图像特征提取与匹配    
//  Author:  www.icvpr.com    
//  Blog  ： http://blog.csdn.net/icvpr      
#include <vector>     
#include <opencv2/opencv.hpp>    
#include "LocalFeature.h"    

using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
	if (argc != 6)
	{
		cout << "wrong usage!" << endl;
		cout << "usage: .exe FAST SIFT BruteForce queryImage trainImage" << endl;
		return -1;
	}

	string detectorType = argv[1];
	string extractorType = argv[2];
	string matchType = argv[3];
	string queryImagePath = argv[4];
	string trainImagePath = argv[5];

	Mat queryImage = imread(queryImagePath, CV_LOAD_IMAGE_GRAYSCALE);
	if (queryImage.empty())
	{
		cout << "read failed" << endl;
		return -1;
	}

	Mat trainImage = imread(trainImagePath, CV_LOAD_IMAGE_GRAYSCALE);
	if (trainImage.empty())
	{
		cout << "read failed" << endl;
		return -1;
	}

	Feature feature(detectorType, extractorType, matchType);

	 //vector<KeyPoint>& queryKeypoints;


	vector queryKeypoints, trainKeypoints;
	feature.detectKeypoints(queryImage, queryKeypoints);
	feature.detectKeypoints(trainImage, trainKeypoints);

	Mat queryDescriptor, trainDescriptor;

	feature.extractDescriptors(queryImage, queryKeypoints, queryDescriptor);
	feature.extractDescriptors(trainImage, trainKeypoints, trainDescriptor);

	vector matches;
	feature.bestMatch(queryDescriptor, trainDescriptor, matches);

	vector<vector> knnmatches;
	feature.knnMatch(queryDescriptor, trainDescriptor, knnmatches, 2);

	Mat outImage;
	feature.saveMatches(queryImage, queryKeypoints, trainImage, trainKeypoints, matches, "../");

	return 0;
}