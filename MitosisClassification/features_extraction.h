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
	for(int i = 0; i < fmat.rows; i++)
		for(int j = 0; j < fmat.cols; j++)
		
			if (fmat.type() == 5)
				features[(i*fmat.cols+j)*dim+findex] = fmat.at<float>(Point(i, j));
			else
				features[(i*fmat.cols+j)*dim+findex] = fmat.at<uchar>(Point(i, j));
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

void extractFeatures(string type, Mat image, int &index,
					 Mat templ, Mat mask, Gerda::Range<float> size, Gerda::Range<float> angle,
//					 HOGDescriptor hog, Size winStride,
					 int dim, vector<float>&features, bool visualize = false) {

	if (visualize) imwrite("features/"+to_string(index)+".png", image);

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
	for(Mat f : bins) {
		addFeature(type, f, findex, index, dim, features, shift_correction);
		if (visualize)
			imwrite((boost::format("features/%d_%d.png") % index % findex).str(), f*255);
	}

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

	/*
	vector<Mat> green_blocks, red_blocks;
	meanOfGridLarge(greenChannel, patch_size, 3, green_blocks);
	for(Mat f : green_blocks) addFeature(type, f, findex, index, dim, features);
	meanOfGridLarge(redChannel, size.max, 3, red_blocks);
	for(Mat f : red_blocks) addFeature(type, f, findex, index, dim, features);
	*/

	// prepare image
	Mat im_hog;
	GaussianBlur(redChannel, im_hog, Size(gauss_patch_size, gauss_patch_size), patch_size/30.0, patch_size/30.0);
	//medianBlur(redChannel, im_hog, 11);
	//if (visualize)
	//	imwrite((boost::format("features/%d_%d.png") % index % findex).str(), im_hog);


	HOGDescriptor hog;
	Size winStride = Size(1, 1);

	int nblocks = 5;
	int hog_patch_size = int(size.max)/nblocks*nblocks;

	hog.nbins = 9;
	
	
	//hog.blockStride = hog.blockSize;
	hog.winSize = Size(hog_patch_size, hog_patch_size);
	hog.cellSize = Size(hog_patch_size/nblocks, hog_patch_size/nblocks);
	hog.blockSize = hog.winSize;
	hog.L2HysThreshold = 0.2;
	hog.winSigma = 100000;

	vector<float> ders;
	hog.compute(im_hog, ders, winStride, Size(0, 0));
	vector<Mat> feat;
	derVec2Mat(image.size(), winStride, hog, ders, feat);
	for(Mat f : feat) addFeature(type, f, findex, index, dim, features);

	if (visualize) {
		vector<float> fvect = getHOGAtPoint(image.size(), winStride, hog, ders,
			Point(image.cols/2, image.rows/2));
		float scaling = 10;
		Mat canvas(image.rows*scaling, image.cols*scaling, CV_8UC3, Scalar(0, 0, 0));
		drawHOGDescriptor(canvas, hog, Point(image.cols/2, image.rows/2), fvect, scaling);
		//cout<<(boost::format("features/%d_%d.png") % index % findex).str()<<endl;
		Mat resized;
		resize(im_hog, resized, Size(image.cols*scaling, image.rows*scaling));
		cvtColor(resized, resized, COLOR_GRAY2RGB);
		imwrite((boost::format("features/%d_%d.png") % index % findex).str(),
			resized+canvas);
	}

	index++;
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


void extractFeaturesNew(vector<Event> events, int &dim, vector<float> &features, bool flip = false) {
	features = vector<float>();
	
	// Load template and mask
	Mat templ, mask;
	templ = imread((boost::format("templates/template%d.png") % 5).str(), 0);
	mask = imread((boost::format("templates/mask%d.png") % 5).str(), 0)/255;
	//mask = Mat::ones(templ.size(), CV_8UC1);

	// Determine number of features
	vector<float> sample1;
	int zero = 0;
	extractFeatures("training", to3Channel(templ), zero, templ, mask,
		Gerda::Range<float>(15,15,5), Gerda::Range<float>(5,5,5), 0, sample1);
	dim = sample1.size();

	// If flipping is on, x4 feature size
	if (flip)
		features = vector<float>(dim*(events.size()*4));
	else
		features = vector<float>(dim*events.size());

	// Index of a training sample
	int idx = 0;

	for(int i = 0; i < events.size(); i++) {

		cout<<i<<'\r'<<flush;

		// Extract patch
		Mat image, patch;
		float size_range = 0.2; int event_size;
		preparePatch(events[i], size_range, image, patch, event_size);

		// Define size andangle range
		float smallest_size = patch.cols*1.0/(1+size_range)*(1-size_range);

		Gerda::Range<float> size(smallest_size, patch.cols+1, (patch.cols-smallest_size)/4);
		Gerda::Range<float> angle(80, 100, 10);

		// Extract features from unflipped
		extractFeatures("training", patch, idx, templ, mask, size, angle, dim, features, true);
		
		//imwrite("positives/"+to_string(idx)+".png", patch);
		
		// Flip
		if (flip) {
			for(int c = -1; c <= 1; c++) {

				Mat flipped;
				cv::flip(patch, flipped, c);
				
				extractFeatures("training", flipped, idx, templ, mask, size, angle, dim, features, true);

			}
		}
	}
	cout<<endl;
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
					//extractFeaturesSimple("testing", rotated, t, templ, mask,
					//	Gerda::Range<float>(s, s, 10), Gerda::Range<float>(0, 0, 10), dim, features);

					extractFeatures("testing", rotated, t, templ, mask,
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
							vector<Point2f> position(1, Point2f(i%rotated.cols, i/rotated.cols));

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
	
	for(int i = 0; i < nfiles; i++) {
		cout<<files[i]<<endl;
		vector<Event> events;
		readAnnotation(files[i], events);
		int dim; vector<float> feat;
		if (labels[i] == 1)
			extractFeaturesNew(events, dim, feat, true);
		else
			extractFeaturesNew(events, dim, feat, true);
		vector<int> lab(feat.size()/dim, labels[i]);
		string new_file = "f"+files[i];
		//createFile(new_file);
		writeToCSV(dim, feat, lab, new_file, false);
	}
}

#endif