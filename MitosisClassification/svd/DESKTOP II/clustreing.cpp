void selectiveClustring(Mat image, Mat mask) {
	int to_sample = countNonZero(mask);
	Mat sampled(1, to_sample, CV_8UC1);
	int sidx = 0;
	for (int i = 0; i < image.rows; i++) {
		for(int j = 0; j < image.cols; j++) {
			if ((int)mask.at<uchar>(i, j) > 0) {
				sampled.at<uchar>(1, sdx) = image.at<uchar>(i, j);
				sdx++;
			}
		}
	}
	// kmeans
	Mat labels;
	sdx = 0;
	Mat vis(image.size(), CV_8UC1, 0);
	for (int i = 0; i < image.rows; i++) {
		for(int j = 0; j < image.cols; j++) {
			if ((int)mask.at<uchar>(i, j) > 0) {
				image.at<uchar>(i, j) = labels.at<uchar>(1, sdx)+1;
				sdx++;
			}
		}
	}
}
void clustVisualize(Mat vis) {
	imwrite("cool.png", vis*50);
}
