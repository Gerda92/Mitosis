#include "includes.h"
#include "file_management.h"
//#include "opencv/"


#ifndef FEATURES_EXTRACTION_H
#define	FEATURES_EXTRACTION_H

void addFeature(float fvalue, int &findex, int index, int dim, vector<float> &features) {
	if (dim == 0) features.push_back(fvalue);
	else features[index*dim+findex] = fvalue;
	findex++;
}

void addFeature(string type, Mat fmat, int &findex, int index, int dim,
				vector<float> &features, int shift_correction = 0) {
	float fvalue = fmat.type() == 5 ? fmat.at<float>(fmat.cols/2, fmat.rows/2) :
		fmat.at<uchar>(fmat.cols/2, fmat.rows/2);
	if (dim == 0) { features.push_back(fvalue); findex++; return; }
	if (type == "training") {
		features[index*dim+findex] = fvalue;

		if (shift_correction > 0) {
			Rect center(fmat.cols/2 - shift_correction, fmat.rows/2 - shift_correction,
				shift_correction*2, shift_correction*2);
			double min, max;
			cv::minMaxLoc(fmat(center), &min, &max);
			features[index*dim+findex] = max;
		}
		findex++; return;
	}
	for(int i = 0; i < fmat.cols; i++)
		for(int j = 0; j < fmat.rows; j++)
			if (fmat.type() == 5)
				features[(i*fmat.rows+j)*dim+findex] = fmat.at<float>(Point(i, j));
			else
				features[(i*fmat.rows+j)*dim+findex] = fmat.at<uchar>(Point(i, j));
	findex++;
}

void preparePatch(Event e, float size_range, Mat &image, Mat &patch, int &event_size) {
	image = e.loadImage();
	float resize_coeff = target_width*1.0/image.cols;

	// resize to standard resolution
	resize(image, image, Size(image.cols*resize_coeff, image.rows*resize_coeff));
	e.resize(resize_coeff);

	event_size = max(e.size, e.dilation*e.size);

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

}

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

void extractFeatures(string type, Mat image, int index, Mat templ, Mat mask, Gerda::Range<float> size, Gerda::Range<float> angle,
					 int dim, vector<float>&features) {

	// start counting features;
	int findex = 0;

	// size for blurring etc., expected event size
	int patch_size = (size.min+size.max)/2;

	Mat redChannel, greenChannel;
	getChannel(image, 2, redChannel);
	getChannel(image, 1, greenChannel);

	int shift_correction = size.min/10;

	/// TEMPLATE MATCHING
	Gerda::Range<float> dil(0.2, 1.1, 0.1);
	vector<Mat> bins;
	templateMatching(redChannel, templ, mask, size, dil, angle, bins, 1);
	for(Mat f : bins) addFeature(type, f, findex, index, dim, features, shift_correction);

	// extract green channel means

	Mat green_mean_center;
	blur(greenChannel, green_mean_center, Size(shift_correction, shift_correction));
	addFeature(type, green_mean_center, findex, index, dim, features);

	// commented to compare to previous function
	Mat red_mean_center;
	blur(redChannel, red_mean_center, Size(shift_correction, shift_correction));
	addFeature(type, red_mean_center, findex, index, dim, features);

	Mat green_over_patch;
	blur(greenChannel, green_over_patch, Size(patch_size, patch_size));
	addFeature(type, green_over_patch, findex, index, dim, features);

	Mat red_over_patch;
	blur(redChannel, red_over_patch, Size(patch_size, patch_size));
	addFeature(type, red_over_patch, findex, index, dim, features);

	int gauss_patch_size = patch_size/2*2+1;
	// Mean of green over a circle
	Mat green_Gaussian1;
	GaussianBlur(greenChannel, green_Gaussian1, Size(gauss_patch_size, gauss_patch_size), patch_size, patch_size);
	addFeature(type, green_Gaussian1, findex, index, dim, features);

	Mat green_Gaussian2;
	GaussianBlur(greenChannel, green_Gaussian2, Size(gauss_patch_size, gauss_patch_size), patch_size/2, patch_size/2);
	addFeature(type, green_Gaussian2, findex, index, dim, features);

	Mat red_Gaussian1;
	GaussianBlur(redChannel, red_Gaussian1, Size(gauss_patch_size, gauss_patch_size), patch_size, patch_size);
	addFeature(type, red_Gaussian1, findex, index, dim, features);

	Mat red_Gaussian2;
	GaussianBlur(redChannel, red_Gaussian2, Size(gauss_patch_size, gauss_patch_size), patch_size/2, patch_size/2);
	addFeature(type, red_Gaussian2, findex, index, dim, features);

	vector<Mat> green_blocks, red_blocks;
	meanOfGridLarge(greenChannel, patch_size, 3, green_blocks);
	for(Mat f : green_blocks) addFeature(type, f, findex, index, dim, features);
	meanOfGridLarge(redChannel, size.max, 3, red_blocks);
	for(Mat f : red_blocks) addFeature(type, f, findex, index, dim, features);

}


void extractFeaturesSimple(string type, Mat image, int index, Mat templ, Mat mask, Gerda::Range<float> size, Gerda::Range<float> angle,
					 int dim, vector<float>&features) {

	_mkdir("features");

	imwrite("features/"+to_string(index)+"im.png", image);

	// start counting features;
	int findex = 0;

	// size for blurring etc., expected event size
	int patch_size = (size.min+size.max)/2;

	Mat redChannel, greenChannel;
	getChannel(image, 2, redChannel);
	getChannel(image, 1, greenChannel);

	int shift_correction = size.min/10;

	/// TEMPLATE MATCHING
	Gerda::Range<float> dil(0.2, 1.1, 0.05);
	vector<Mat> bins;
	templateMatching(redChannel, templ, mask, size, dil, angle, bins, dil.n());
	for(Mat f : bins) {
		addFeature(type, f, findex, index, dim, features, shift_correction);
		//imwrite("features/"+to_string(index)+".png", f*255);
	}

	Mat green_mean_center;
	//blur(greenChannel, green_mean_center, Size(shift_correction, shift_correction));
	//addFeature(type, green_mean_center, findex, index, dim, features);
	//imwrite("features/"+to_string(index)+"green.png", green_mean_center);
}


void extractFeaturesNew(vector<Event> events, int &dim, vector<float> &features) {
	features = vector<float>();
	vector<float> sample1;

	Mat templ, mask;
	templ = imread((boost::format("templates/template%d.png") % 1).str(), 0);
	//mask = imread((boost::format("templates/mask%d.png") % 5).str(), 0)/255;
	mask = Mat::ones(templ.size(), CV_8UC1);

	extractFeaturesSimple("training", events[0].loadImage(), 0, templ, mask,
		Gerda::Range<float>(15,15,5), Gerda::Range<float>(5,5,5), 0, sample1);
	dim = sample1.size();
	features = vector<float>(dim*events.size());
	for(int i = 0; i < events.size(); i++) {
		cout<<i<<endl;
		Mat image, patch;
		float size_range = 0.2; int event_size;
		preparePatch(events[i], size_range, image, patch, event_size);

		float smallest_size = patch.cols*1.0/(1+size_range)*(1-size_range);

		Gerda::Range<float> size(smallest_size, patch.cols+1, (patch.cols-smallest_size)/4);
		Gerda::Range<float> angle(80, 100, 10);
		extractFeaturesSimple("training", patch, i, templ, mask, size, angle, dim, features);
	}
}

void traverseMovie(Movie movie) {
	int slice = 5;
	vector<Event> events;
	_mkdir("movie");
	readAnnotation("E:/IDP/annotations/140820_031915/2015-03-19_FVF-mTmG_Wnt3a_9_t1-140_all_data.csv", events);
	for(int t = 1; t <= movie.nframes; t++) {
		Mat image = movie.loadImage(slice, t);
		vector<Event> slice_events = findAssociatedEvents(events, t, slice);
		for(int i = 0; i < slice_events.size(); i++) {
			slice_events[i].draw(image);
		}
		imwrite("movie/"+movie.getName(slice, t), image);
	}
}

void drawEvent(Mat &image, Event e) {
	double radius = e.size/2.0;
	circle(image, e.getPoint(),
		radius, Scalar(255, 0, 255), 1);
	putText(image, to_string(e.eventID), e.getPoint(), 1, 1, Scalar(255, 255, 255));
	
	int x = cos((270-e.orientation)/180.0*M_PI)*radius;
	int y = sin((270-e.orientation)/180.0*M_PI)*radius;
	line(image, e.getPoint(),
		e.getPoint() + Point(x, y), Scalar(255, 0, 255), 1);	
}

template<typename C>
void quickEventExtract(Movie movie, int dim, C classifier, string new_folder) {

	//string new_folder = "svm classifed simple/";
	_mkdir(new_folder.c_str());

	// Get events of the movie
	vector<Event> events;
	readAnnotation("mothers.csv", events);
	events = getEventsFromMovie(events, movie.id);

	string csv_path = new_folder+"class.csv";
	Event().writeHeaderToFileExtended(csv_path);
	string all_csv_path = new_folder+"all_class.csv";
	Event().writeHeaderToFileExtended(all_csv_path);

	Mat templ, mask;
	templ = imread((boost::format("templates/template%d.png") % 1).str(), 0);
	mask = Mat::ones(templ.size(), CV_8UC1);
	//mask = imread((boost::format("templates/mask%d.png") % 5).str(), 0)/255;

	// interate over slices and frames
	for(int z = 4; z <= 6; z++) {
		for(int t = 1; t <= movie.nframes; t++) {

			// event id within an image
			int id = 0;

			Mat image = movie.loadImage(z, t);

			float standard_resize_coeff = target_width*1.0/image.cols;
			float resize_coeff = 200.0/image.cols;

			// resize to standard resolution
			resize(image, image, Size(image.cols*resize_coeff, image.rows*resize_coeff));

			// if there are positives, skip
			//vector<Event> slice_events = findAssociatedEvents(events, t, z);
			//if (slice_events.size() > 0) continue;

			vector<Event> detected; vector<float> prob;
			
			Gerda::Range<float> size(30, 100, 10);
			Gerda::Range<float> angle(0, 180, 10);

			size.resize(resize_coeff/standard_resize_coeff);

			for(int a = angle.min; a <= angle.max; a+=angle.interval) {

				// rotate
				Mat rotated;
				rotate(image, a, rotated);

				for(int s = size.min; s <= size.max; s+=size.interval) {

					cout <<int(((a-angle.min)/angle.interval*size.n()+(s-size.min)/size.interval)
						*100.0/size.n()/angle.n()) << '\r' << flush;
				
					// Features extraction
					vector<float> features(rotated.cols*rotated.rows*dim);
					extractFeaturesSimple("testing", rotated, t, templ, mask,
						Gerda::Range<float>(s, s, 10), Gerda::Range<float>(0, 0, 10), dim, features);

					// Classification
					vector<float> result;
					Data data(features, dim, vector<int>());
					result = applyClassifier(classifier, data);

					// Retrieving positive
					for(int i = 0; i < result.size(); i++) {
						if (result[i] >= 0.5) {
							//Point position(i/image.rows, i%image.rows);
							//circle(image, position, s/2, Scalar(255, 255, 255));

							// Event position
							vector<Point2f> position(1, Point2f(i/rotated.rows, i%rotated.rows));

							// Rotating back
							Mat rot_mat = getRotationMatrix2D(Point(rotated.cols/2, rotated.rows/2), -a, 1.0);
							transform(position, position, rot_mat);
							Point2f translation(rotated.cols/2 - image.cols/2, rotated.rows/2 - image.rows/2);
							Point2f loc = position[0]-translation;

							//circle(image, position, s/2, Scalar(255, 255, 255));

							int orientation = a >= 90 ? -a + 270 : -a + 90;

							Event e(id, movie.id, "Mother", Point3f(loc.x, loc.y, 14), z, t, s, 1, orientation);

							detected.push_back(e);
							prob.push_back(result[i]);

							e.resize(1.0/resize_coeff);
							e.writeEventToFileExtended(all_csv_path);

							id++;
						}
					}

				}

				// end of f
			}

			cout<<"# detected: "<<detected.size()<<endl;

			vector<Event> reduced = reduceEvents(detected, 7, prob);

			cout<<"# reduced: "<<reduced.size()<<endl;

			image = movie.loadImage(z, t);

			//vector<Event> reduced = detected;
			for(int i = 0; i < reduced.size(); i++) {

				reduced[i].resize(1.0/resize_coeff);

				//circle(image, reduced[i].getPoint(),
				//	reduced[i].size/2, Scalar(255, 0, 255), 1);
				//putText(image, to_string(reduced[i].eventID), reduced[i].getPoint(), 1, 1, Scalar(255, 255, 255));

				drawEvent(image, reduced[i]);

				reduced[i].writeEventToFileExtended(csv_path);
			}
			imwrite(new_folder+movie.getName(z, t), image);

			// end of s
		}

		// end of t traversing
	}

	// end of z traversing
}

void reviseAnnotation(Movie movie, string annotation, string new_folder) {

	vector<Event> events;
	//_mkdir("movie");

	string new_annotation = new_folder + "new_class.csv";
	Event().writeHeaderToFileExtended(new_annotation);

	readAnnotation(annotation, events);

	for(int z = 5; z < 6; z++) {
		for(int t = 31; t <= movie.nframes; t++) {

			Mat image = movie.loadImage(z, t);
			vector<Event> slice_events = findAssociatedEvents(events, t, z);
			for(int i = 0; i < slice_events.size(); i++) {
				cout<<slice_events[i].eventID<<" "<<slice_events[i].orientation<<endl;
				slice_events[i].orientation =
					slice_events[i].orientation >= 90 ?
						-slice_events[i].orientation + 270 :
						-slice_events[i].orientation + 90;
				cout<<slice_events[i].eventID<<" "<<slice_events[i].orientation<<endl;
				//slice_events[i].orientation = 40;
				circle(image, slice_events[i].getPoint(),
					slice_events[i].size/2.0, Scalar(255, 0, 255), 1);
				putText(image, to_string(slice_events[i].eventID), slice_events[i].getPoint(), 1, 1, Scalar(255, 255, 255));
				double radius = slice_events[i].size/2.0;
				int x = cos((270-slice_events[i].orientation)/180.0*M_PI)*radius;
				int y = sin((270-slice_events[i].orientation)/180.0*M_PI)*radius;
				line(image, slice_events[i].getPoint(),
					slice_events[i].getPoint() + Point(x, y), Scalar(255, 0, 255), 1);
				slice_events[i].writeEventToFileExtended(new_annotation);
			}
			
			//imshow("Im", image);
			imwrite(new_folder+movie.getName(z, t), image);
			/*
			cout<<"List positive (comma delimited): ";
			//waitKey(1);

			string text;
			cin>>text;
			if (text != "n") {
				vector<string> elements;
				split(text, ',', elements);
				for(int i = 0; i < elements.size(); i++) {
					int neg_id = atoi(elements[i].c_str());
					Event e = getWithID(slice_events, neg_id, "Mother");
					cout<<"Writing event "<<e.eventID<<endl;
					e.writeEventToFileExtended(new_annotation);

				}
			}
			*/

		}
	}

}

template<class F>
void extractCells(Mat image, int dim, const Forest<F, MyAggregator> forest) {
	for(int s = 20; s <= 70; s+=5) {
		for(int a = 0; a <= 180; a+=10) {
			vector<float> features;
			extractFeatures(image, s, a, dim, features);
			vector<MyAggregator> result;
			Data data(features, dim, vector<int>());
			Test<FeatureResponse>(forest, data, result);
			float count = 0;
			for(int i = 0; i < result.size(); i++) {
				if (result[i].GetProbability(1) >= 0.5) {
					circle(image, Point(i/image.rows, i%image.rows), s/2, Scalar(255, 255, 255));
					//circle(image, Point(i%image.rows, i/image.rows), s/2, Scalar(255, 255, 0));
				}
			}

		}
		//imshow("x", image);
		//waitKey(0);
		imwrite("res.png", image);
	}
	imwrite("res.png", image);
	//waitKey(0);

}

void writeFeaturesToFile(int nfiles, string files[], int labels[], string filename) {
	createFile(filename);
	for(int i = 0; i < nfiles; i++) {
		vector<Event> events;
		readAnnotation(files[i], events);
		int dim; vector<float> feat;
		extractFeaturesNew(events, dim, feat);
		vector<int> lab(feat.size()/dim, labels[i]);
		writeToCSV(dim, feat, lab, filename);
	}
}

#endif