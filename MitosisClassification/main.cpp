#include "includes.h"
#include "Event.h"
#include "file_management.h";
#include "features_extraction.h"
#include "extract_contour.h"
#include "training.h"

//#include "RandomForest.h"
//#include "Classification.h"
//#include "Visualize.h"

#include <Eigen/Dense>
#include <Eigen/SVD>

using Eigen::Vector2f;
using Eigen::MatrixXf;

void template_matching_test();

int main(int argc, char** argv) {

	//reviseAnnotation(Movie::getMovies()[0], "classified tm/class.csv", "classified tm/");

	//template_matching_test();
	//return 0;
		
	/*
	Mat image = Movie::getMovies()[0].loadImage(5, 3);
	Mat rotated;
	rotate(image, 40, rotated);
	Point2f point(200, 200);

	vector<Point2f> position(1, point);
	Mat rot_mat = getRotationMatrix2D(Point(rotated.cols/2, rotated.rows/2), -40, 1.0);
	transform(position, position, rot_mat);
	Point2f translation(rotated.cols/2 - image.cols/2, rotated.rows/2 - image.rows/2);
	circle(image, position[0] - translation, 3, Scalar(255,255,255), -1);
	circle(rotated, point, 3, Scalar(255,255,255), -1);
	
	Point2f center(rotated.cols/2, rotated.rows/2);
	circle(image, center - translation, 3, Scalar(255,255,255), -1);
	circle(rotated, center, 3, Scalar(255,255,255), -1);
	imshow("Image", image);
	imshow("Rotated", rotated);
	waitKey(0);

	return 0;
	*/
	string annotations[] = {"positive_mothers.csv", "handcrafted_negative.csv", "random_mothers.csv"};
	//string annotations[] = {"positive_mothers.csv"};
	//string annotations[] = {"random_mothers.csv"};
	int labs[] = {1, 0, 0};
	//int labs[] = {1};
	//writeFeaturesToFile(sizeof(labs)/sizeof(int), annotations, labs, "simple.csv");
	//return 0;
	
	int dim; vector<float> features; vector<int> labels; 
	readFromCSV(dim, features, labels, "simple.csv");

	Data data(features, dim, labels);

	Ptr<SVM> svm;
	//trainSVM(data, svm, 100000, 0.00003);

	/*
	
	//double cs[] = {0.001, 0.01, 0.1, 1, 100, 1000, 10000};
	double cs[] = {100000, 10000000, 10000000};
	double gammas[] = {0.00001, 0.01};
	//double gammas[] = {10000, 100000, 1000000, 10000000, 10000000};
	for (int i = 0; i < sizeof(cs)/sizeof(double); i++) {
		for(double j = gammas[0]; j < gammas[1]; j+=0.00001) {
			cout<<cs[i]<<" "<<j<<endl;
			trainSVM(data, svm, cs[i], j);
		}
	}
	*/

	/*
	string forest_file = "forest.txt";
	//quickTrain(data, forest_file);
	std::ifstream is (forest_file, std::ifstream::binary);

	auto_ptr<Forest<FeatureResponse, MyAggregator>> forest(new Forest<FeatureResponse, MyAggregator>);
	forest = Forest<FeatureResponse, MyAggregator>::Deserialize(is);

	cout<<Classifier<Forest<FeatureResponse, MyAggregator>>(*forest).getType()<<endl;
	*/

	//quickEventExtract(Movie::getMovies()[0], data.dimension_, svm);

	vector<Movie> movies = Movie::getMovies();
	for(int i = 0; i < movies.size(); i++) {
		quickEventExtract(movies[i], data.dimension_, 0.5, movies[i].series_name+"/");
	}
	cout<<"Finish."<<endl;
	getchar();
}

void template_matching_test() {

	string dir = "template matching 1 compare";

	_mkdir(dir.c_str());

	Movie movie = Movie::getMovies()[0];

	for (int i = 31; i < 50; i++) {
		int z = 5; int t = i;
		Mat image = movie.loadImage(z, i);
		//GaussianBlur(image, image, Size(21, 21), 2, 2);
		resize(image, image, Size(image.cols*1.0/image.rows*200, 200));
		//resize(image, image, Size(image.cols/2, image.rows/2));
		getChannel(image, 2, image);
		imwrite(dir+"/"+movie.getName(z,t), image);
		Mat templ, mask, map;
		templ = imread("templates/template1.png", 0);
		//mask = imread("templates/mask5.png", 0)/255;
		mask = Mat::ones(templ.size(), CV_8UC1);
		//resize(mask, mask, templ.size());

		Gerda::Range<float> size(50/2, 100/2, 5);
		Gerda::Range<float> dil(0.2, 1, 0.05);
		Gerda::Range<float> angle(0, 180, 10);

		vector<Mat> bins;
		templateMatching(image, templ, mask, size, dil, angle, bins, dil.n());

		Mat vis(image.size(), CV_8UC1, Scalar(0));
		for(int i = 0; i < image.cols; i++) {
			for(int j = 0; j < image.rows; j++) {
				if (bins[0].at<float>(j, i) > 0.5) {
					circle(vis, Point(i, j), 1, Scalar(255, 255, 255), -1);
				}
			}
		}

		imwrite(dir+"/im"+movie.getName(z,t), image);
		imwrite(dir+"/map"+movie.getName(z,t), bins[0]*255);

		vector<Mat> bgr;
		bgr.push_back(Mat(image.size(), CV_8UC1, Scalar(0)));
		bgr.push_back(vis); bgr.push_back(image);
		Mat merged;
		merge(bgr, merged);
		imwrite(dir+"/mer"+movie.getName(z,t), merged);
		//imshow(to_string(t), merged);
		//waitKey(0);

	}

}