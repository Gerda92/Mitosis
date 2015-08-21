#include "includes.h"

#ifndef TEMPLATE_MATCHING_H
#define	TEMPLATE_MATCHING_H

#define _USE_MATH_DEFINES
#include <math.h>

/*
*	AFFINE TRANSFORMATIONS OF MAT
*/

void rotate(Mat &src, double angle, Mat &dst) {
	int newsize = max(src.rows, src.cols);
	Mat ext(newsize, newsize, src.type(), Scalar(0));
	src.copyTo(ext(Rect((newsize-src.cols)/2, (newsize-src.rows)/2, src.cols, src.rows)));
	Point2f src_center(newsize/2.0F, newsize/2.0F);
	Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
	warpAffine(ext, dst, rot_mat, Size(newsize, newsize));
}

void transform(Mat src, double new_size, double dilation, double angle, Mat &dst) {
	resize(src, dst, Size(new_size, new_size*dilation));
	rotate(dst, angle, dst);
}

/*
*	TEMPLATE MATCHING
*/

void matchMaskedTemplate(Mat I, Mat T, Mat M, Mat &result) {
	//cout<<"Template size: "<<T.size()<<endl;
	I.convertTo(I, CV_32FC1);
	T.convertTo(T, CV_32FC1);
	M.convertTo(M, CV_32FC1);
	//M = Mat::ones(T.size(), CV_32FC1);

	double mu_t = mean(T)[0];

	Mat mu_i;
	blur(I, mu_i, T.size());

	Mat T_prime = (T-mu_t).mul(M);

	//cout<<I.type()<<" "<<T_prime.type()<<endl;
	//cout<<(T-mu_t).mul(M).size()<<endl;

	Mat numerator; // sum_x',y' (T(x',y') - mu_t)*M(x',y')*(I(x+x',y+y') - mu_i(x,y))
	cv::matchTemplate(I, (T-mu_t).mul(M), numerator, TM_CCORR);

	// Cut borders
	Rect ROI = Rect(T.cols/2, T.rows/2, I.cols-T.cols+1, I.rows-T.rows+1);

	numerator = numerator - sum((T-mu_t).mul(M))[0]*mu_i(ROI);
	/*
	Mat check_num; 
	cv::matchTemplate(I, T, check_num, TM_CCOEFF);
	Mat diff = numerator-check_num;
	//cout<<"First check! Should be 0: "<<mean(abs(diff))<<endl;

	//imshow("my", toVisible(numerator));
	//imshow("check", toVisible(check_num));
	//waitKey(0);
	*/
	Mat T_denom, I_denom(T.size(), CV_32FC1);
	sqrt(sum((T-mu_t).mul(T-mu_t)), T_denom);

	// I' = I - mu_i	
	Mat I_prime = I - mu_i;
	
	Mat I_prime_squared(T.size(), CV_32FC1);
	multiply(I_prime, I_prime, I_prime_squared);

	// UNSTABLE!!!
	filter2D(I_prime_squared, I_denom, -1, M);
	//imshow("I'", toVisible(I_denom));
	Mat neg;
	inRange(I_denom, -std::numeric_limits<float>::max(), 1, neg);
	neg.convertTo(neg, CV_32FC1);
	// Whatever, at places with zero mean nothing can be
	I_denom = I_denom - I_denom.mul(neg/255) + neg/255*1000;

	//Mat small;
	//inRange(I_denom, 0, 1, small);
	//cout<<-std::numeric_limits<float>::max()<<endl;
	//imshow("small", toVisible(small));
	//small.convertTo(small, CV_32FC1);
	//I_denom = I_denom - small*std::numeric_limits<float>::max();

	//imshow("neg", toVisible(neg(ROI)));
	//imshow("my", toVisible(I_denom(ROI)));
	//blur(I_prime_squared, I_denom, T.size());
	//I_denom = I_denom*T.cols*T.rows;
	
	//imshow("check", toVisible(I_denom));
	
	sqrt(I_denom, I_denom);
	//imshow("sqrt", toVisible(I_denom(ROI)));
	//imshow("numerator", toVisible(numerator));
	//imshow("denom", toVisible(I_denom(ROI)));

	Mat res = numerator/T_denom/I_denom(ROI);	

	Mat neg_cov, pos_cov;
	inRange(res, -std::numeric_limits<float>::max(), -1.0001, neg_cov);
	inRange(res, 1.0001, std::numeric_limits<float>::max(), pos_cov);
	//imshow("neg_cov", toVisible(neg_cov));
	//imshow("pos_cov", toVisible(pos_cov));

	pos_cov.convertTo(pos_cov, CV_32FC1);
	res = res - res.mul(pos_cov);

	//imshow("res", res);

	//waitKey(0);
	//Mat check_res;
	//cv::matchTemplate(I, T, check_res, TM_CCOEFF_NORMED);

	//diff = res-check_res;
	//cout<<"Should be 0: "<<mean(abs(diff))<<endl;
	result = res;
	//imshow("my", toVisible(res));
	//imshow("check", toVisible(check_res));
	//waitKey(0);

	/*
	Mat masked_templ;
	multiply(templ, mask, masked_templ);
	//imshow("fddsf", masked_templ);
	//waitKey(0);
	//cout<<masked_templ<<endl;
	Mat ccorr(image.size(), CV_32FC1);
	cv::matchTemplate(image, masked_templ, ccorr, TM_CCORR);
	// ccorr = sum_x',y' T(x',y')M(x',y')I(x+x',y+y')
	//check
	Mat ccorr_check;
	cv::matchTemplate(image, templ, ccorr_check, TM_CCORR);
	cout<<"First check! Should be 0: "<<sum(abs(ccorr_check - ccorr))<<endl;


	Mat ccorr_mask(image.size(), CV_32FC1);
	cv::matchTemplate(image, mask, ccorr_mask, TM_CCORR);
	//cout<<mask*1.0/mask.cols/mask.rows<<endl;
	//imshow("mean", toVisible(ccorr_mask*1.0/templ.cols/templ.rows));
	//waitKey(0);
	// ccorr_mask = sum_x',y' M(x',y')I(x+x',y+y')

	double mut = mean(templ)[0];
	Mat mui;
	blur(image, mui, templ.size());

	mui = mui(Rect(templ.cols/2, templ.rows/2, image.cols-templ.cols+1, image.rows-templ.rows+1));
	//mui = mui(Rect(0, 0, image.cols-templ.cols+1, image.rows-templ.rows+1));

	//imshow("mean2", toVisible(mui));
	

	// check
	Mat diff;
	diff = ccorr_mask*1.0/templ.cols/templ.rows - mui;
	cout<<"Should be 0: "<<sum(abs(diff))<<endl;
	waitKey(0);
	// sum_x',y' (T(x',y') - mu_t)*M(x',y')*I(x+x',y+y')
	//Mat masked_unnorm = ccorr - mut*ccorr_mask -
	//	mui*sum(templ)[0] + mui*mut;

	// check
	Mat check1;
	cv::matchTemplate(image, masked_templ-mut*mask, check1, TM_CCORR);
	cout<<"Should be 0: "<<sum(abs(masked_templ-templ))<<endl;
	//cout<<masked_templ-(mut*mask)<<endl<<endl;
	//cout<<templ-mut<<endl;
	diff = masked_templ - (mut*mask) - (templ-mut);
	//cout<<abs(diff)<<endl;
	cout<<"Should be 0: "<<sum(abs(diff))<<endl;
	diff = check1 - (ccorr - mut*ccorr_mask);
	cout<<"Should be 0: "<<sum(abs(diff))<<endl;

	// sum_x',y' (T(x',y') - mu_t)*M(x',y')*I(x+x',y+y')
	Mat masked_unnorm = check1 -
		mui*sum(masked_templ)[0] + mui*mut*countNonZero(mask);

	Mat squared_masked_templ, masked_sum;
	// (M(x',y')*(T(x',y')-mu_t))^2
	multiply(masked_templ-mut*mask, masked_templ-mut*mask, squared_masked_templ);

	// I' = I - mu_i	
	Mat I_prime = image(Rect(templ.cols/2, templ.rows/2, image.cols-templ.cols+1, image.rows-templ.rows+1)) - mui;
	
	Mat I_prime_squared;
	multiply(I_prime, I_prime, I_prime_squared);

	filter2D(I_prime_squared, masked_sum, CV_32FC1, mask);
	//diff = masked_sum/countNonZero(mask) - mui.mul;
	//cout<<"Should be 0: "<<sum(abs(diff))<<endl;
	Mat sqrt_masked_sum;
	sqrt(masked_sum, sqrt_masked_sum);
	divide(masked_unnorm/sqrt(sum(squared_masked_templ)[0]), sqrt_masked_sum, result);
	*/
}

namespace Gerda {

template <typename T> 
struct Range {
	T min, max, interval;
	Range() {};
	Range(T mini, T maxi, T intervali) : min(mini), max(maxi), interval(intervali) {};
	bool includes_max() {
		return fmod(max-min,interval) == 0;
	}
	int n() {
		return (max - min)/interval + 1;
	}
	void resize(float coeff) {
		min = min*coeff; max = max*coeff; interval = interval*coeff;
	}

};

}

// For fining best template configuration
void templateMatching(Mat &image, Mat templ, Mat mask,
					  Gerda::Range<float> size, Gerda::Range<float> dil, Gerda::Range<float> angle,
					  vector<Mat> &bins, int ballspbin,
					  Point &maxPoint, float &max_size, float &max_dil, float &max_angle,
					  Rect searchRegion = Rect()) {

	// number of balls to be spread accross bins
	int nballs = dil.n();

	int nbins = nballs/ballspbin + (nballs%ballspbin == 0 ? 0 : 1);
	bins = vector<Mat>(nbins);
	for(int i = 0; i < nbins; i++) {
		bins[i] = Mat::zeros(image.size(), CV_32FC1);
	}

	double maxCorr = 0;

	if (searchRegion.area() == 0) // ROI is not provided
		searchRegion = Rect(0, 0, image.cols, image.rows);

	// to use in rearching max locations
	Mat searchMask = Mat::zeros(image.size(), CV_8UC1);
	searchMask(searchRegion) = Mat::ones(searchRegion.size(), CV_8UC1)*255;

	for (float s = size.min; s <= size.max; s += size.interval) { // scale
		for (float d = dil.min; d <= dil.max; d += dil.interval) { // dilation
			if (max(s*d, s) > min(image.cols, image.rows)) continue; // template should not be larger than image
			for (float a = angle.min; a <= angle.max; a += angle.interval) {

				// resized, stretched and rotated template and mask
				Mat to_slide, new_mask;
				transform(templ, s, d, a, to_slide);
				transform(mask, s, d, a, new_mask);

				//imwrite(str(boost::format("res/templ_%d_%d_%d.png") % s % (d*s) % a), to_slide);
				//imwrite(str(boost::format("res/mask_%d_%d_%d.png") % s % (d*s) % a), new_mask*255);

				// if there is no mask (all is taken into an account),
				// do standard template matching, otherwise apply my method
				Mat result;
				if (countNonZero(mask) == mask.cols*mask.rows)
					cv::matchTemplate(image, to_slide, result, TM_CCOEFF_NORMED);
				else
					matchMaskedTemplate(image, to_slide, new_mask, result);

				// sort into appropriate bin
				int ball_i = (s - dil.min)/dil.interval; // index of ball
				int dest_bin = ball_i/ballspbin; // destination bin
				
				Mat bigger(image.size(), CV_32FC1, Scalar(0));
				result.copyTo(bigger(Rect(to_slide.cols/2, to_slide.rows/2, result.cols, result.rows)));
				max(bins[dest_bin], bigger, bins[dest_bin]);

				// identify best match
				double minVal; double maxVal; Point minLoc; Point maxLoc;
				//Mat patch = bigger(searchRegion).clone();
				//cout<<patch.size()<<endl;
				minMaxLoc(bigger, &minVal, &maxVal, &minLoc, &maxLoc, searchMask);
				if (maxVal > maxCorr) {
					maxCorr = maxVal;
					maxPoint = maxLoc;//Point(searchRegion.x, searchRegion.y) + maxLoc;
					max_size = s; max_dil = d; max_angle = a;
				}
	
			}
		}
	}
}

// For extracting features based on template matching
void templateMatching(Mat &image, Mat templ, Mat mask,
					  Gerda::Range<float> size, Gerda::Range<float> dil, Gerda::Range<float> angle,
					  vector<Mat> &bins, int ballspbin) {

	// number of balls to be spread accross bins
	int nballs = dil.n();

	//int nbins = nballs/ballspbin + (nballs%ballspbin == 0 ? 0 : 1);
	int nbins = nballs/ballspbin;
	bins = vector<Mat>(nbins);
	for(int i = 0; i < nbins; i++) {
		bins[i] = Mat::zeros(image.size(), CV_32FC1);
	}

	for (float s = size.min; s <= size.max; s += size.interval) { // scale
		for (float d = dil.min; d < dil.max; d += dil.interval) { // dilation
			if (max(s*d, s) >= min(image.cols, image.rows)) {
				//cout<<"template should not be larger than image"<<endl;
				continue; // template should not be larger than image
			}

			for (float a = angle.min; a <= angle.max; a += angle.interval) {

				// resized, stretched and rotated template and mask
				Mat to_slide, new_mask;
				transform(templ, s, d, a, to_slide);
				transform(mask, s, d, a, new_mask);

				//imwrite(str(boost::format("res/templ_%d_%d_%d.png") % s % (d*s) % a), to_slide);
				//imwrite(str(boost::format("res/mask_%d_%d_%d.png") % s % (d*s) % a), new_mask*255);

				// if there is no mask (all is taken into an account),
				// do standard template matching, otherwise apply my method
				Mat result;
				if (countNonZero(mask) == mask.cols*mask.rows)
					cv::matchTemplate(image, to_slide, result, TM_CCOEFF_NORMED);
				else
					matchMaskedTemplate(image, to_slide, new_mask, result);

				// sort into appropriate bin
				//cout<<d<<endl;
				int ball_i = round((d - dil.min)/dil.interval); // index of ball
				int dest_bin = ball_i/ballspbin; // destination bin
				
				Mat bigger(image.size(), CV_32FC1, Scalar(0));
				result.copyTo(bigger(Rect(to_slide.cols/2, to_slide.rows/2, result.cols, result.rows)));
				max(bins[dest_bin], bigger, bins[dest_bin]);

			}
		}
	}
}

void getMaxBinAndLoc(vector<Mat> bins, int &bin_i, Point &max) {

	double maxCorr = 0;
	for(int i = 0; i < bins.size(); i++) {
		double minVal; double maxVal; Point minLoc; Point maxLoc;
		minMaxLoc(bins[i], &minVal, &maxVal, &minLoc, &maxLoc, Mat());
		if (maxCorr < maxVal) {
			maxCorr = maxVal;
			bin_i = i;
			max = maxLoc;
		}
	}
}

/*
*	EXTRACTING PATCHES FROM IMAGES
*/

// With image augumentation
void safePatchExtraction(Mat image, Rect ROI, Mat &patch) {
	Mat bigger = Mat::zeros(image.rows+ROI.height*2, image.cols+ROI.width*2, image.type());
	image.copyTo(bigger(Rect(ROI.width, ROI.height, image.cols, image.rows)));
	patch = bigger(Rect(ROI.x+ROI.width, ROI.y+ROI.height, ROI.width, ROI.height));
}

// Cuts only what it has to, center may shift
bool safePatchExtraction(Size im_size, int x, int y, int w, int h, Rect &frame) {
	bool resp = false;
	if (x < 0) { frame.x = 0; }
	else { frame.x = x; resp = true; }
	if (y < 0) frame.y = 0;
	else { frame.y = y; resp = true; }
	if (w + frame.x > im_size.width) frame.width = im_size.width-frame.x;
	else { frame.width = w; resp = true; }
	if (h + frame.y > im_size.height) frame.height = im_size.height-frame.y;
	else { frame.height = h; resp = true; }
	return resp;
}

// Leaves center where it was
void safePatchExtractionSymmetric(Size im_size, Rect &frame) {
	int centerx = frame.x + frame.width/2; int centery = frame.y + frame.height/2;
	frame.width = min(frame.width/2, min(centerx, im_size.width - centerx))*2;
	frame.height = min(frame.height/2, min(centery, im_size.height - centery))*2;
	frame.x = centerx - frame.width/2;
	frame.y = centery - frame.height/2;
}

Mat oneChannelToThree(Mat image) {
	cv::Mat out;
	cv::Mat in[] = {image, image, image};
	cv::merge(in, 3, out);
	return out;
}

void meanOfGrid(Mat image, int channel, int n, vector<float> &means) {
	means = vector<float>(n*n);
	Mat ch;
	getChannel(image, channel, ch);
	int ncols = image.cols/n; int nrows = image.rows/n;
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			means[i*n+j] = mean(ch(Rect(i*ncols, j*nrows, ncols, nrows)))[0];
		}
	}
}

void plotMeandOfGrid(Size size, int n, vector<float> means, Mat &image) {
	image = Mat::zeros(size, CV_8UC1);
	int ncols = image.cols/n; int nrows = image.rows/n;
	Mat piece = Mat::ones(Size(ncols, nrows), CV_8UC1);
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			Mat col_piece = piece*means[i*n+j];
			col_piece.copyTo(image(Rect(i*ncols, j*nrows, ncols, nrows)));
		}
	}
}

Mat shift(Mat src, Point p) {
	Mat dst = Mat::zeros(src.size(), src.type());
	Rect frame;
	safePatchExtraction(src.size(), p.x, p.y, src.cols-abs(p.x), src.rows-abs(p.y), frame);
	src(Rect(frame.x-p.x, frame.y-p.y, frame.width, frame.height)).copyTo(dst(frame));
	return dst;
}

void meanOfGridLarge(Mat image, int patch_size, int n, vector<Mat> &means) {
	means = vector<Mat>(n*n);
	int block = patch_size/n;
	Mat mean;
	blur(image, mean, Size(block, block));
	//imshow("b", mean);
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			int shiftx = block*(i-n/2);
			int shifty = block*(j-n/2);
			means[i*n+j] = shift(mean, Point(shiftx, shifty));
			//imshow("a", means[i*n+j]);
			//waitKey(0);
		}
	}
}

void copyToCenter(Mat &b, Mat f) {
	f.copyTo(b(Rect(b.cols/2-f.cols/2, b.rows/2-f.rows/2, f.cols, f.rows)));
}

/*
*	READING AND WRITING FEATURES TO FILE
*/

void writeToCSV(int dim, vector<float> feat, vector<int> lab, string filename, bool cont = true) {
    ofstream f;
	if (!cont) f.open(filename.c_str(), ios::out);
	f.close();
    f.open(filename.c_str(), ios::app);
	for(int i = 0; i < feat.size(); i++) {
		if (i%dim == dim - 1) {
			f << feat[i] << "," << lab[i/dim] << endl;
		} else {
			f << feat[i] << ",";
		}
	}
	f.close();
}

void readFromCSV(int &dim, vector<float> &feat, vector<int> &lab, string filename) {
	ifstream fin;
    fin.open(filename, ios::in);

	string text;

	feat.reserve(200);

    while (std::getline(fin, text)) {

        vector<string> elements = split(text, ',');
		dim = elements.size()-1;
		for(int i = 0; i < dim; i++) {
			feat.push_back(atof(elements[i].c_str()));
		}
		lab.push_back(atoi(elements[dim].c_str()));
	}
}

void vectorToMat(int dim, vector<float> v, Mat &m) {
	m = Mat(v.size()/dim, dim, CV_8UC1, Scalar(0));
	for(int i = 0; i < v.size(); i++) {
		m.at<uchar>(i/dim, i%dim) = v[i]*255;
	}
	imshow("df", m);
	waitKey(0);
}


#endif