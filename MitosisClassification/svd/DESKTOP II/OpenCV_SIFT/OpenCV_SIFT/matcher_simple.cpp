
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "direct.h"

#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	// Load the two images.
	// 'query' and 'train' are the notation used by the parameters in the 'match' function.
	// It seems backwards from how I'm applying it--the first image is where I've
	// isolated the object I'm looking for, and the second is the image I want to locate that
	// object in.
	string pic = "5_ar";
	Mat queryImg = imread("E:/IDP/software/source/template1.png", 0);
    Mat trainImg = imread("E:/IDP/software/source/Division events/FVF-mTmG_Wnt3a_9a/" + pic + ".png", 0);
    
	// Verify the images loaded successfully.
    if(queryImg.empty() || trainImg.empty())
    {
        printf("Can't read one of the images\n");
        return -1;
    }

    // Detect keypoints in both images.
    SiftFeatureDetector detector(400);
    vector<KeyPoint> queryKeypoints, trainKeypoints;
    detector.detect(queryImg, queryKeypoints);
    detector.detect(trainImg, trainKeypoints);

	// Print how many keypoints were found in each image.
	printf("Found %d and %d keypoints.\n", queryKeypoints.size(), trainKeypoints.size());

    // Compute the SIFT feature descriptors for the keypoints.
	// Multiple features can be extracted from a single keypoint, so the result is a
	// matrix where row 'i' is the list of features for keypoint 'i'.
    SiftDescriptorExtractor extractor;
    Mat queryDescriptors, trainDescriptors;
    extractor.compute(queryImg, queryKeypoints, queryDescriptors);
    extractor.compute(trainImg, trainKeypoints, trainDescriptors);

	// Print some statistics on the matrices returned.
	Size size = queryDescriptors.size();
	printf("Query descriptors height: %d, width: %d, area: %d, non-zero: %d\n", 
		   size.height, size.width, size.area(), countNonZero(queryDescriptors));
	
	size = trainDescriptors.size();
	printf("Train descriptors height: %d, width: %d, area: %d, non-zero: %d\n", 
		   size.height, size.width, size.area(), countNonZero(trainDescriptors));

    // For each of the descriptors in 'queryDescriptors', find the closest 
	// matching descriptor in 'trainDescriptors' (performs an exhaustive search).
	// This seems to only return as many matches as there are keypoints. For each
	// keypoint in 'query', it must return the descriptor which most closesly matches a
	// a descriptor in 'train'?
    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> matches;
    matcher.knnMatch(queryDescriptors, trainDescriptors, matches, 2);
	vector<DMatch> good_matches, first_best, second_best;
	for (int i = 0; i < matches.size(); i++) {
		if (matches[i][0].distance < 0.7*matches[i][1].distance)
			good_matches.push_back(matches[i][0]);
		first_best.push_back(matches[i][0]);
		second_best.push_back(matches[i][1]);
	}

	printf("Found %d matches.\n", matches.size());

    // Draw the results. Displays the images side by side, with colored circles at
	// each keypoint, and lines connecting the matching keypoints between the two 
	// images.
	namedWindow("matches", WINDOW_NORMAL);
	namedWindow("keypoints", WINDOW_NORMAL);
	namedWindow("filtered", WINDOW_NORMAL);
	_mkdir(pic.c_str());
    Mat img_matches;
	drawKeypoints(trainImg, trainKeypoints, img_matches);
	imshow("keypoints", img_matches);
	imwrite(pic + "/keypoints.png", img_matches);
    drawMatches(queryImg, queryKeypoints, trainImg, trainKeypoints, first_best, img_matches);
    imshow("matches", img_matches);
	imwrite(pic + "/first_matches.png", img_matches);
    drawMatches(queryImg, queryKeypoints, trainImg, trainKeypoints, second_best, img_matches);
    imshow("matches", img_matches);
	imwrite(pic + "/second_matches.png", img_matches);
    drawMatches(queryImg, queryKeypoints, trainImg, trainKeypoints, good_matches, img_matches);
    imshow("filtered", img_matches);
	imwrite(pic + "/good_matches.png", img_matches);
	
    //waitKey(0);

    return 0;
}
