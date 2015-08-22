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


void extractFeatures(Event e, Mat templ, Mat mask, int index, int dim, vector<float> &features) {

	cout<<index<<endl;
	
	//if (index != 10) return;

	Mat image = e.loadImage();
	float resize_coeff = target_width*1.0/image.cols;

	// resize to standard resolution
	resize(image, image, Size(image.cols*resize_coeff, image.rows*resize_coeff));
	e.resize(resize_coeff);

	int event_size = max(e.size, e.dilation*e.size);

	// maximual size of template to apply
	//int max_size = event_size*1.2;
	float size_range = 0.1;

	Mat patch;

	// initial patch extraction, before rotation
	int max_size1 = min(image.rows, image.cols)/2;
	safePatchExtraction(image, Rect(e.coordinates.x - max_size1/2, e.coordinates.y - max_size1/2,
		max_size1, max_size1), patch);

	rotate(patch, -e.orientation, patch);

	// size after rotation
	int max_size2 = event_size*(1+size_range)+1;
	safePatchExtraction(patch, Rect(patch.cols/2 - max_size2/2, patch.rows/2 - max_size2/2,
		max_size2, max_size2), patch);

	int final_res = 50;
	resize(patch, patch, Size(final_res, final_res));

	imwrite("res/"+to_string(index)+".png", patch);

	// Start extracting features
	// feature counter
	int findex = 0;

	// extract template matching results

	//float smallest_size = final_res/1.2*0.8;
	//Gerda::Range<float> size(smallest_size, final_res+1, (final_res-smallest_size)/4);

	float smallest_size = final_res/(1+size_range)*(1-size_range);
	//cout<<smallest_size<<" "<<final_res<<endl;
	Gerda::Range<float> size(smallest_size, final_res+1, (final_res-smallest_size)/2);
	Gerda::Range<float> dil(0.2, 1.1, 0.1);
	Gerda::Range<float> angle(90, 90, 10);
	
	vector<Mat> bins;

	Mat redChannel;
	getChannel(patch, 2, redChannel);

	templateMatching(redChannel, templ, mask, size, dil, angle, bins, 1);

	int shift_correction = final_res/10;
	Rect center(patch.cols/2 - shift_correction, patch.rows/2 - shift_correction,
		shift_correction*2, shift_correction*2);

	Mat overlay(patch.size(), CV_8UC1, Scalar(0));
	rectangle(overlay, center, Scalar(255, 255, 255));
	imwrite("res/"+to_string(index)+"_center.png", redChannel+overlay);

	float glob_max = -1; int maxi;
	for(int i = 0; i < bins.size(); i++) {

		imwrite("res/"+to_string(index)+"_"+to_string(i)+".png", bins[i]*255);
		double min, max;
		//cout<<bins[i](center)<<endl;
		cv::minMaxLoc(bins[i](center), &min, &max);
		if (glob_max < max) { glob_max = max; maxi = i;}
		//features.push_back(max);
		addFeature(max, findex, index, dim, features);
	}

	Mat overlay2(patch.size(), CV_8UC3, Scalar(0, 0, 0));
	rectangle(overlay2, center, Scalar(0, 255, 0));
	Mat best;
	drawSingleChannel(bins[maxi]*255, overlay2, best);
	imwrite("res/"+to_string(index)+"_"+to_string(maxi)+".png", best);

	float green_mean_center = mean(patch(center))[1];
	//features.push_back(green_mean_center);
	addFeature(green_mean_center, findex, index, dim, features);

	Scalar means_over_patch = mean(patch);
	// green
	addFeature(means_over_patch[1], findex, index, dim, features);
	// red
	addFeature(means_over_patch[2], findex, index, dim, features);

	// Mean of green inside cell. Taking best shape of template-matching
	float best_dil = dil.min+dil.interval*maxi;
	Mat transformed_mask;
	transform(mask, final_res/(1+size_range)*0.8, best_dil, 90, transformed_mask);
	Mat patch_mask = Mat::zeros(patch.size(), CV_8UC1);
	copyToCenter(patch_mask, transformed_mask);
	imwrite("res/"+to_string(index)+"_best_mask.png", patch_mask*255);

	//cout<<oneChannelToThree(patch_mask).type()<<endl;
	//Mat test = oneChannelToThree(patch_mask);
	//test.convertTo(test, CV_8U);
	Scalar inside_means = mean(patch, patch_mask);
	//cout<<"Inside mean: "<<inside_means<<endl;
	// green
	imwrite("res/"+to_string(index)+"_"+to_string(findex)+".png", patch_mask*inside_means[1]*5);
	addFeature(inside_means[1], findex, index, dim, features);
	// red
	imwrite("res/"+to_string(index)+"_"+to_string(findex)+".png", patch_mask*inside_means[2]*5);
	addFeature(inside_means[2], findex, index, dim, features);
	

	// Grid means green
	vector<float> piece_means_green, piece_means_red;
	int npieces = 3;
	meanOfGrid(patch, 1, npieces, piece_means_green);
	Mat pieces_vis;
	plotMeandOfGrid(patch.size(), npieces, piece_means_green, pieces_vis);
	for(int i = 0; i < npieces*npieces; i++) addFeature(piece_means_green[i], findex, index, dim, features);
	imwrite("res/"+to_string(index)+"_"+to_string(findex)+".png", pieces_vis);

	// Grid means red
	meanOfGrid(patch, 2, npieces, piece_means_red);
	plotMeandOfGrid(patch.size(), npieces, piece_means_red, pieces_vis);
	for(int i = 0; i < npieces*npieces; i++) addFeature(piece_means_red[i], findex, index, dim, features);
	imwrite("res/"+to_string(index)+"_"+to_string(findex)+".png", pieces_vis);

}

void extractFeatures(vector<Event> events, int &dim, vector<float> &features) {
	features = vector<float>();
	vector<float> sample1;

	Mat templ, mask;
	templ = imread((boost::format("templates/template%d.png") % 5).str(), 0);
	mask = imread((boost::format("templates/mask%d.png") % 5).str(), 0)/255;
	//mask = Mat::ones(templ.size(), CV_8UC1);

	extractFeatures(events[0], templ, mask, 0, 0, sample1);
	dim = sample1.size();
	features = vector<float>(dim*events.size());
	for(int i = 0; i < events.size(); i++) {
		extractFeatures(events[i], templ, mask, i, dim, features);
	}
}


#endif