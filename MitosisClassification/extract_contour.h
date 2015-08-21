#include "includes.h"
#include <math.h>
#include <numeric>
#include "spline.h"

#include "Event.h"

using namespace tk;

#ifndef EXTRACT_COUNTOUR_H
#define	EXTRACT_COUNTOUR_H

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out << std::setprecision(n) << a_value;
    return out.str();
}

struct RefPoint { // to sort points w.r.t. distance from a reference point and calculate angle
	Point2f c;
	RefPoint(Point2f a): c(a) {};
	public:
		double angle(Point2f p1, Point2f p3) { // clockwise vector between two vectors
			double x1 = p1.x - c.x; double y1 = p1.y - c.y;
			double x2 = p3.x - c.x; double y2 = p3.y - c.y;
			double dot = x1*x2 + y1*y2;	// dot product
			double det = x1*y2 - y1*x2;	// determinant
			return 180 / 3.14159265358979323846 * atan2(det, dot);	// atan2(y, x) or atan2(sin, cos)
		}
		double dot(Point2f v1, Point2f v2) {
			return v1.x*v2.x+v1.y*v2.y;
		}
		double angle2(Point2f p1, Point2f p3) {
			Point2f v1 = p1-c; Point2f v2 = p3-c;
			double rawangle = 180 / 3.14159265358979323846 * abs(acos(dot(v1, v2)/norm(v1)/norm(v2)));
			return rawangle > 90 ? 180 - rawangle : rawangle;
		}
		bool operator () (Point2f a, Point2f b) {
			return norm(a - c) > norm(b - c);
		}
};

int findClosestPoint(vector<Point2f> curve, Point2f p) {
	return max_element(curve.begin(), curve.end(), RefPoint(Point2f(p.x, p.y))) - curve.begin();
}

Point findNthClosestPoint(vector<Point2f> curve, Point p, int n = 0) {
	nth_element(curve.begin(), curve.begin() + n, curve.end(), RefPoint(Point(p.x, p.y)));
	return *(curve.begin() + n);
}

vector<vector<Point2f>> extractRawCountour(Mat image, Mat overlay = Mat(), Mat &resmask = Mat()) {

	Mat tofill = image.clone();
	int rrnd = rand()%100;
	//imwrite("Events/orig"+to_string(rrnd)+".png", image);
	// To prevent floodFill() leaking through crypt border
	GaussianBlur(tofill, tofill, Size(5, 5), 3);
	medianBlur(tofill, tofill, 31);
	//imwrite("Events/blur"+to_string(rrnd)+".png", tofill);
	Mat mask(image.rows + 2, image.cols + 2, CV_8UC1, Scalar(0,0,0));

	// Choose a black corner as a seed for floodFill()
	int offset = 20;
	Point corners[] = {Point(offset, offset), Point(image.cols - 1 - offset, offset),
		Point(image.cols - 1 - offset, image.rows - 1 - offset), Point(offset, image.rows - 1 - offset)};

	int crn = -1;

	for (int i = 0; i < 4; i++) {
		if ((int)tofill.at<uchar>(corners[i].y, corners[i].x) == 0) {
			crn = i;
			// FloodFill algorithm expanding to black pixels
			floodFill(tofill, mask, corners[crn], Scalar(255, 255, 255),
				0, Scalar(), Scalar(), 4);
		}
		//circle(overlay, corners[i], 3, Scalar(255, 255, 255), -1, 8);
		//putText(overlay, to_string((int)tofill.at<uchar>(corners[i].y, corners[i].x)),corners[i] + Point(-5, -8),FONT_HERSHEY_COMPLEX_SMALL, .8, cvScalar(255, 255, 255), 1, false);			
	}
	resmask = mask.clone();
	//imshow("MMask", mask*255);
	
	//imwrite("Events/tofill"+to_string(rrnd)+".png", tofill);
	//imwrite("Events/mask"+to_string(rrnd)+".png", mask*255);
	if (crn == -1) {
		cout<<"None of the corners are black. The crypt surface cannot be extracted.\n";
		return vector<vector<Point2f>>();
	} else {
		//cout<<"Corner: "<<corners[crn].y<<" "<<corners[crn].x<<" "
		//	<<(int)tofill.at<uchar>(corners[crn].y, corners[crn].x)<<endl;
	}

	vector<Mat> contours;
	// Extracts external contour
	findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	//drawContours(overlay, contours, -1, Scalar(255, 255, 255));
	cout<<"Number of contours: "<<contours.size()<<endl;

	vector<vector<Point2f>> curve(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		vector<Mat> points;
		split(contours[i], points);
		vector<int> x; vector<int> y;
		points[0].reshape(1, 1).row(0).copyTo(x); points[1].reshape(1, 1).row(0).copyTo(y);
		for (int j = 0; j < x.size(); j++) curve[i].push_back(Point2f(x[j], y[j]));
	}
	return curve;
}

vector<Point2f> purifyContour(vector<Point2f> contour, Mat image, Mat overlay = Mat()) {
	struct BorderPoint {
		Mat image;
		BorderPoint(Mat img): image(img) {}
		public:
		bool isBorderPoint(Point p) {
			return p.x == 1 || p.y == 1 || p.x == image.cols || p.y == image.rows;
		}
	};
	vector<int> breakpoints;
	for (int i = 0; i < contour.size(); i++) {
		if (!BorderPoint(image).isBorderPoint(contour[i]) && breakpoints.size()%2 == 0)
			breakpoints.push_back(i);
		if (BorderPoint(image).isBorderPoint(contour[i]) && breakpoints.size()%2 == 1)
			breakpoints.push_back(i-1);
	}
	if (breakpoints.size()%2 == 1)
		breakpoints.push_back(contour.size()-1);
	
	int maxnpoints = -1; int maxi = -1;
	for (int i = 0; i < breakpoints.size(); i+=2) {
		//circle(overlay, contour[breakpoints[i]], 3, Scalar(255, 255, 255), -1, 8);
		//circle(overlay, contour[breakpoints[i+1]], 3, Scalar(255, 255, 255), -1, 8);
		if (breakpoints[i+1] - breakpoints[i] > maxnpoints) {
			maxnpoints = breakpoints[i+1] - breakpoints[i]; maxi = i;
		}
	}
	return vector<Point2f>(contour.begin() + breakpoints[maxi], contour.begin() + breakpoints[maxi + 1]);
}

vector<vector<Point2f>> purifyContours(vector<vector<Point2f>> contours, Mat image, Mat overlay = Mat()) {
	for_each(contours.begin(), contours.end(), [image, overlay](vector<Point2f> &c){ c = purifyContour(c, image, overlay); });
	return contours;
}

vector<Point2f> sampleFromContour(vector<Point2f> curve, int interval) {
	vector<Point2f> newcurve;
	for (int i = 0; i < curve.size(); i+=interval) {
		newcurve.push_back(curve[i]);
	}
	return newcurve;
}

void drawContour(vector<Point2f> contour, Mat overlay) {
	for (int i = 0; i < contour.size(); i++) {
		circle(overlay, contour[i], 3, Scalar(255, 255, 255), -1, 8);
		//putText(overlay, to_string(i), contour[i] + Point2f(10, 10),FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(255, 255, 255), 1);
	} 
	cout<<"Number of contour points: "<<contour.size()<<endl;
}

vector<Point2f> interpolate(vector<double> t, spline sx, spline sy) {
	vector<Point2f> result;
	transform(t.begin(), t.end(), back_inserter(result),
		[sx, sy] (double tt) {return Point2f(sx(tt), sy(tt));});
	return result;
}

vector<double> getRange(double range1, double range2) {
	vector<double> t(int((range2 - range1)*100) + 1);
	for (double i = 0; i < t.size(); i++) t[i] = range1 + i/100.0;
	return t;
}

Point2f getVector(double slope, double length = 1) {
	return Point2f(length/sqrt(1 + slope*slope), length*slope/sqrt(1 + slope*slope));
}

void drawTangent(Point2f p, double slope, Mat overlay) {
	double length = 50;
	Point2f shiftVector = getVector(slope, length);
	line(overlay, p + shiftVector, p - shiftVector, Scalar(255, 255, 255), 1);
}

double calcAngle(vector<vector<Point2f>> contours, Point2f daught1, Point2f daught2, Mat overlay, Scalar color = Scalar(255, 0, 255)) {

	line(overlay, daught1 , daught2, Scalar(255, 0, 0), 2);
	Point2f p = (daught1 + daught2)/2.0;

	vector<int> maxis; vector<Point2f> maxvs;
	transform(contours.begin(), contours.end(), back_inserter(maxis),
		[p] (vector<Point2f> contour) {return findClosestPoint(contour, p);});
	transform(maxis.begin(), maxis.end(), contours.begin(), back_inserter(maxvs),
		[] (int i, vector<Point2f> contour) {
			//cout<<contour[i]<<" ";
			return contour[i];
	});

	int maxcontour = findClosestPoint(maxvs, p); int maxp = maxis[maxcontour];
	vector<Point2f> angleCont = contours[maxcontour];

	//cout<<endl<<"Maxp "<<contours[maxcontour][maxp]<<endl;

	int maxp2;
	if (maxp == angleCont.size() - 1) maxp2 = maxp - 1;
	else {
		if (maxp == 0) maxp2 = maxp + 1;
		else maxp2 = RefPoint(p)(angleCont[maxp+1],
			angleCont[maxp-1]) ? maxp-1 : maxp+1;
	}

	//cout<<endl<<"Maxp2 "<<contours[maxcontour][maxp2]<<endl;
	circle(overlay, p, 3, color, -1, 8);
	circle(overlay, angleCont[maxp], 3, color, -1, 8);
	circle(overlay, angleCont[maxp2], 3, color, -1, 8);

	// Separate angleCont to x and y
	vector<double> x(angleCont.size()); vector<double> y(angleCont.size());
	for (int i = 0; i < angleCont.size(); i++) {
		x[i] = angleCont[i].x; y[i] = angleCont[i].y;
	}

	spline sx, sy;
	vector<double> t(angleCont.size());
	iota(t.begin(), t.end(), 0);
	sx.set_points(t, x);
	sy.set_points(t, y);
	vector<double> newt = getRange(0, t.size() - 1);
	vector<Point2f> interp = interpolate(newt, sx, sy);
	for (int i = 0; i < interp.size(); i++) {
		circle(overlay, interp[i], 0.5, Scalar(0, 0, 255), -1, 8);
	}

	int range1 = min(maxp, maxp2); int range2 = max(maxp, maxp2);

	vector<double> tint = getRange(range1, range2);

	vector<Point2f> interp2 = interpolate(tint, sx, sy);
	for (int i = 0; i < interp2.size(); i++) {
		circle(overlay, interp2[i], 0.5, color, -1, 8);
	}

	int closest = findClosestPoint(interp2, p);
	double tpoint = closest*0.01+range1;
	//cout<<tpoint<<endl;
	circle(overlay, interp2[closest], 3, color, -1, 8);
	double slope = sy.derivative(tpoint)/sx.derivative(tpoint);
	drawTangent(interp2[closest], slope, overlay);

	double angle = RefPoint(p).angle2(daught1, p + getVector(slope));

	putText(overlay, to_string_with_precision(angle, 4), interp2[closest] + Point2f(10, 10),FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255), 1, false);

	return angle;
}

void extractContour(Mat image, Mat &mask, vector<Point2f> &contour, int points = -1) {
	Mat overlay;
	vector<vector<Point2f>> contours = extractRawCountour(image, overlay, mask);
	contour = purifyContour(contours[0], image, overlay);
	int interval = (int)sqrt(image.rows*image.cols)/10;
	contour = sampleFromContour(contour, interval);
}

Mat extractContour(Mat image, vector<Event> d1, vector<Event> d2, Mat &mask) {
		Mat overlay = Mat::zeros(image.size(), CV_8UC3);

		vector<vector<Point2f>> contour = extractRawCountour(image, overlay, mask);
		contour = purifyContours(contour, image, overlay);
		int interval = (int)sqrt(image.rows*image.cols)/10;
		vector<vector<Point2f>> sampledCont;
		for(vector<Point2f> c : contour) {
			vector<Point2f> samples = sampleFromContour(c, interval);
			drawContour(samples, overlay);
			sampledCont.push_back(samples);
		};
		/*
		// Testing with some points
		Point p[] = {Point2f(0, 0), Point2f(300, 300), Point (200, 300), Point2f(300, 200),
			Point2f(200, 200), Point (100, 200)};
		Scalar c[] = {Scalar(255, 255, 0), Scalar(0, 255, 255), Scalar(255, 0, 255),
			Scalar(0, 0, 255), Scalar(255, 0, 0), Scalar(0, 255, 0)};

		for (int i = 0; i < 6; i++) {
			calcAngle(sampledCont, p[i], overlay, c[i]);
		}
		*/
		for (int i = 0; i < d1.size() && i < d2.size(); i++) {
			Point3f p1 = d1[i].getCoordinates(); Point3f p2 = d2[i].getCoordinates();
			Point2f daught1 = Point2f(p1.x, p1.y); Point2f daught2 = Point2f(p2.x, p2.y);

			double tangentSlope = calcAngle(sampledCont, daught1, daught2, overlay);
			
			
		}

		//vector<Point2f> samples = sampleFromContour(curve, interval);
		

		

		struct X { // 
			Point c;
			X(Point a): c(a) {};
			public:
				float angle(Point p1, Point p3) {
					float x1 = p1.x - c.x; float y1 = p1.y - c.y;
					float x2 = p3.x - c.x; float y2 = p3.y - c.y;
					float dot = x1*x2 + y1*y2;	// dot product
					float det = x1*y2 - y1*x2;	// determinant
					return 180 / 3.14159265 * atan2(det, dot);	// atan2(y, x) or atan2(sin, cos)
				}
				bool operator () (Point a, Point b) {
					return norm(a - c) > norm(b - c);
				}
		};
		/*
		for (int i = 0; i < d1.size() && i < d2.size(); i++) {
			Point3f p1 = d1[i].getCoordinates(); Point3f p2 = d2[i].getCoordinates();
			Point daught1 = Point(p1.x, p1.y); Point daught2 = Point(p2.x, p2.y);
			line(contour, Point(p1.x, p1.y), Point(p2.x, p2.y), Scalar(255, 0, 0), 2);
			Point3f center = (p1 + p2)/2.0;
			//int maxa = max_element(curve.begin(), curve.end(), X(Point(p1.x, p1.y))) - curve.begin();
			Point a = *max_element(curve.begin(), curve.end(), X(Point(p1.x, p1.y)));
			Point b = *max_element(curve.begin(), curve.end(), X(Point(center.x, center.y)));
			Point c = *max_element(curve.begin(), curve.end(), X(Point(p2.x, p2.y)));
			line(contour, a, c, Scalar(0, 255, 0), 2);
			circle(contour, a, 3, Scalar(0, 255, 255), -1, 8);
			circle(contour, b, 3, Scalar(255, 0, 0), -1, 8);
			circle(contour, c, 3, Scalar(0, 0, 255), -1, 8);

			float angle = abs(X(c).angle(a, daught1 - daught2 + c));
			putText(contour, to_string_with_precision(angle, 4), c + Point(10, 10),FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(255, 255, 255), 1, false);
		}

		*/

	return overlay;
}

void extractCryptMask(Mat image, Mat &mask) {

	// To prevent floodFill() leaking through crypt border
	GaussianBlur(image, image, Size(5, 5), 3);
	medianBlur(image, image, 31);

	mask = Mat(image.rows + 2, image.cols + 2, CV_8UC1, Scalar(0,0,0));

	// Choose a black corner as a seed for floodFill()
	int offset = 20;
	Point corners[] = {Point(offset, offset), Point(image.cols - 1 - offset, offset),
		Point(image.cols - 1 - offset, image.rows - 1 - offset), Point(offset, image.rows - 1 - offset)};

	int crn = -1;

	for (int i = 0; i < 4; i++) {
		if ((int)image.at<uchar>(corners[i].y, corners[i].x) == 0) {
			crn = i;
			// FloodFill algorithm expanding to black pixels
			floodFill(image, mask, corners[crn], Scalar(255, 255, 255),
				0, Scalar(), Scalar(), 4);
		}
	}
	//imshow("mask.png", mask*255);
	//waitKey(0);
	bitwise_not(mask*255, mask);
}

void blurMask(Mat src, Mat &dst, Size kernel_size, Mat mask) {
	Mat image32f, mask32f;
	src.convertTo(image32f, CV_32FC1);
	mask.convertTo(mask32f, CV_32FC1);
	image32f = image32f.mul(mask32f);
	Mat blurred, blurred_mask;
	blur(image32f, blurred, kernel_size);
	blur(mask32f, blurred_mask, kernel_size);
	dst = blurred/blurred_mask;
	dst.convertTo(dst, CV_8UC1);
}

#endif