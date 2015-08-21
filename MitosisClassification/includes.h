#ifndef INCLUDES_H
#define	INCLUDES_H

#define _USE_MATH_DEFINES

#include <stdio.h>
#include <cmath>
#include <string>
#include <iostream>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <fstream>
#include <vector>
#include <time.h>
#include <math.h>

#include <opencv2/opencv.hpp> 
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

#include <boost/format.hpp>
/*
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
*/

#include "RandomForest.h"

#include <direct.h>

using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace boost;
//using namespace boost::filesystem;
using namespace MicrosoftResearch::Cambridge::Sherwood;
/*
*
* HELPER FUNCTIONS
*
*/

double round(double val);

std::vector<std::string> &split(std::string &s, char delim, std::vector<std::string> &elems);

std::vector<std::string> split(std::string &s, char delim);

void getChannel(Mat &src, int channel_index, Mat &channel);

Mat toVisible(Mat img);

// Draw with color on one-channel matrix
void drawSingleChannel(Mat image, Mat canvas, Mat &result);

void createFile(string name);

// width to resize every image to
static const int target_width = 300;

#endif