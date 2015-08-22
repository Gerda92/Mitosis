#include "includes.h"

// split string with delimiter

std::vector<std::string> &split(std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

std::vector<std::string> split(std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

// get a channel of an image

void getChannel(Mat &src, int channel_index, Mat &channel) {
	vector<Mat> bgr;
	split(src, bgr);
	channel = bgr[channel_index];
}

double round(double val) {    
    return floor(val + 0.5);
}

Mat toVisible(Mat img) {
	double min, max;
	minMaxLoc(img, &min, &max);
	cout<<"Min max "<<min<<" "<<max<<endl;
	Mat raw = (img-min)/(max-min)*255;
	Mat res;
	raw.convertTo(res, CV_8UC1);
	return res;
}

Mat to3Channel(Mat image) {
	Mat color[] = {image, image.clone(), image.clone()};
			
	Mat merged;
	merge(color, 3, merged);
	merged.convertTo(merged, CV_8UC3);
	return merged;
}

void drawSingleChannel(Mat image, Mat canvas, Mat &result) {
	Mat color[] = {image, image.clone(), image.clone()};
			
	Mat merged;
	merge(color, 3, merged);
	merged.convertTo(merged, CV_8UC3);

	result = merged + canvas;
}

void createFile(string name) {
	ofstream myfile;
	myfile.open (name);
	myfile.close();
}