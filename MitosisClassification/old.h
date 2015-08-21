#ifndef OLD_H
#define	OLD_H

#include "includes.h"
#include "image_loading.h"
#include "extract_contour.h"
#include "features.h"

// code to test different resolutions
void resolutions_test() {
	string folder_name = "E:/IDP/movies";

	string series_name = "FVF-mTmG_Wnt3a_9";

	Mat image;
	int z = 5; int t = 30;
	load_image(image, folder_name, series_name, z, t);
	vector<Mat> bgr;
	split(image, bgr);
	image = bgr[2];
	float factors[] = {500, 400, 300, 200, 100, 50};
	for (int i = 0; i < sizeof(factors)/sizeof(float); i++) {
		Mat resized;
		resize(image, resized, Size(image.cols*1.0/image.rows*factors[i], factors[i]));

		cout<<boost::format("%.2f.png") % factors[i]<<endl;

		imwrite(str(boost::format("%.2f.png") % factors[i]), resized);
	}
}

// code to make pattern matching over many time frames
void template_matching_test() {


	string folder_name = "E:/IDP/movies";

	string series_name = "FVF-mTmG_Wnt3a_9";

	for (int i = 1; i < 50; i++) {
		Mat image;
		int z = 5; int t = i;
		loadImage(image, folder_name, series_name, z, t);
		resize(image, image, Size(image.cols*1.0/image.rows*300, 300));
		//resize(image, image, Size(image.cols/2, image.rows/2));
		getChannel(image, 2, image);
		imwrite("res/orig.png", image);
		Mat templ, mask, map;
		templ = imread("templates/template5.png", 0);
		mask = imread("templates/mask5.png", 0)/255;
		//mask = Mat::ones(templ.size(), CV_8UC1);
		Mat vis = templateMatching(image, templ, mask, map);
		imwrite("res/matching.png", vis);
		imwrite("res/map.png", map);
		vector<Mat> bgr;
		bgr.push_back(Mat(image.size(), CV_8UC1, Scalar(0)));
		bgr.push_back(vis); bgr.push_back(image);
		Mat merged;
		merge(bgr, merged);
		imwrite("res/merged.png", merged);
		imshow(to_string(t), merged);
		waitKey(0);
	}

}

// code to test SVD
void SVDtest() {
	string folder_name = "E:/IDP/movies";

	string series_name = "FVF-mTmG_Wnt3a_9";

	_mkdir("svd");

	for (int i = 1; i < 50; i++) {
		Mat image;
		int z = 5; int t = i;
		loadImage(image, folder_name + "/" + series_name, series_name, z, t);
		Mat mask; vector<Point2f> contour;
		extractContour(image, mask, contour);
		drawContour(contour, image);

		MatrixXf m(2, contour.size());
		float meanx = 0; float meany = 0;
		for (int j = 0; j < contour.size(); j++) {
			m(0, j) = contour[j].x; meanx += contour[j].x;
			m(1, j) = contour[j].y; meany += contour[j].y;
		}

		meanx /= contour.size();
		meany /= contour.size();

		Eigen::JacobiSVD<MatrixXf> svd(m.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);

		Vector2f c1 = svd.matrixV().col(0);
		Vector2f c2 = svd.matrixV().col(1);

		cout<<c1<<endl;
		cout<<c2<<endl;

		cout<<c1(0)/c1(1)<<endl;

		drawTangent(Point(meanx, meany), c1(0)/c1(1), image);
		drawTangent(Point(meanx, meany), c2(0)/c2(1), image);

		imwrite("svd/"+to_string(i)+".png", image);
		//waitKey(0);
	}
}

// recognizing mothers
void recognizing_mothers() {
	string base_path = "E:/IDP/annotations";
	string annotations[] = {"140820_031915", "145845_031915", "151605_042415",
		"160145_031915", "181040_031915"};
	vector<string> a(annotations, annotations+5);
	_mkdir("events");

	ifstream headers;
	headers.open(base_path+"/headers.txt");

	for (int i = 0; i < a.size(); i++) {
		string head_file;
		getline(headers, head_file);
		cout<<head_file;
		if (i != 1) continue;

		string folder_name, series_name, file_format;
		int first, last, interval, n_frames;
		vector<Event> events;
		loadData(base_path + "/" + head_file, folder_name, series_name, first, last,
			interval, n_frames, file_format, events);
		extractEvents(folder_name, series_name, events, "events/"+a[i], base_path+"/"+a[i]+".csv");
	}
}

// calculate mean and std of positive
void testMoviesEvents() {
	string base_path = "E:/IDP";

	vector<Movie> movies = Movie::getMovies();
	for(int i = 0; i < movies.size(); i++) {
		cout<<movies[i].getPath(1, 1)<<endl;
	}
	
	
	vector<Event> events;
	readAnnotation("E:/IDP/annotations/positive samples/training.csv", events);
	float mean = 0;
	float std = 0;
	for(int i = 0; i < events.size(); i++) {
		float resolution_coeff = 300.0/events[i].getMovie().resolution.width;
		//float tsize = events[i].size*events[i].dilation*resolution_coeff;
		float tsize = events[i].dilation;
		mean += tsize;
		std += tsize*tsize;
		cout<<events[i].eventID<<" "<<tsize<<endl;
	}

	mean /= events.size();
	std = sqrt(std/events.size() - mean*mean);
	cout<<mean<<" "<<std<<endl;
}

void forestTrainingAndVisualization() {
	float arr[] = {1, 1,  2, 3,  1, 3,  -1, -3,  -5, -1,  1, -1, -2, 4};
	vector<float> feat(arr, arr+14);
	int arrl[] = {0, 0, 0, 1, 1, 1, 1};
	vector<int> lab(arrl, arrl+7);

	Data data(feat, 2, lab);

    TrainingParameters trainingParameters;
    trainingParameters.MaxDecisionLevels = 10;
    trainingParameters.NumberOfCandidateFeatures = 2;
    trainingParameters.NumberOfCandidateThresholdsPerFeature = 3;
    trainingParameters.NumberOfTrees = 10;
    trainingParameters.Verbose = true;

    // Load training data for a 2D density estimation problem.
    //std::auto_ptr<Data> trainingData = std::auto_ptr<Data> (&data);
	
	Random random;

	FeatureResponse fResp(2);
		
	TrainingContext<FeatureResponse> classificationContext(2, &fResp);

	std::auto_ptr<Forest<FeatureResponse, MyAggregator> > forest 
		= ForestTrainer<FeatureResponse, MyAggregator>::TrainForest (
			random, trainingParameters, classificationContext, data);
	
	vector<MyAggregator> result;
	std::auto_ptr<Data> testingData = Data::Generate2dGrid(make_pair(-5.0, 5.0), 400, make_pair(-5.0, 5.0), 400);
	Mat canvas(200, 200, CV_8UC3, Scalar(0,0,0));
	Test<FeatureResponse>(*forest, *testingData, result);

	for(int i = 0; i < (*testingData).Count(); i++) {
		circle(canvas, Point(((*testingData).GetDataPoint(i)[0]+5)*20, ((*testingData).GetDataPoint(i)[1]+5)*20), 0.5, Scalar(255, 255*result[i].GetProbability(1), 0), -1);		
	}
	
	for(int i = 0; i < data.Count(); i++) {
		circle(canvas, Point((data.GetDataPoint(i)[0]+5)*20, (data.GetDataPoint(i)[1]+5)*20), 4, Scalar(0, 0, 0), -1);
		circle(canvas, Point((data.GetDataPoint(i)[0]+5)*20, (data.GetDataPoint(i)[1]+5)*20), 3, Scalar(255, 255*lab[i], 0), -1);
	}
	
	imwrite("forest.png", canvas);
}

	/*
	vector<Event> events;
	separateMothersFromDaughters("E:/IDP/annotations/positive samples/training.csv",
		"positive_mothers.csv", "positive_daughters.csv");
	
	readAnnotation("E:/IDP/annotations/positive samples/training.csv", events);
	vector<float> features;
	extractFeatures(events, features);
	*/
	//extractRandom(Movie::getMovies(), 100, 10, "mothers.csv", "daughters.csv");
	//std::getchar();

	/*
	string base_path = "E:/IDP/annotations/best annotation";
	string annotations[] = {"140820_031915", "145845_031915", "151605_042415",
		"160145_031915", "181040_031915"};
	vector<string> a(annotations, annotations+5);
	_mkdir("events");

	ifstream headers;
	headers.open(base_path+"/headers.txt");

	
	for (int i = 0; i < a.size(); i++) {
		string head_file;
		getline(headers, head_file);
		cout<<head_file;

		string folder_name, series_name, file_format;
		int first, last, interval, n_frames;
		vector<Event> events;
		loadData(base_path + "/" + head_file, folder_name, series_name, first, last,
			interval, n_frames, file_format, events);
		recognizeSizes(folder_name, series_name, events, "events/"+a[i], a[i]+".csv");
	}
	*/


#endif