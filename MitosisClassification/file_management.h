#include "includes.h"
#include "Event.h"
#include "template_matching.h"
#include "extract_contour.h"

#ifndef FILE_MANAGEMENT_H
#define	FILE_MANAGEMENT_H

/*
*	ANNOTATION READING
*/

void readAnnotation(string path, vector<Event> &events) {

	ifstream fin;
    fin.open(path, ios::in);

	string text;

    // read header
    getline(fin, text);

    // read data

    while (std::getline(fin, text)) {

        vector<string> elements = split(text, ',');
        int eventID = atoi(elements[0].c_str());
        string cell_type = elements[1];
		Point3f coordinates;
        coordinates.x = atof(elements[2].c_str());
        coordinates.y = atof(elements[3].c_str());
        coordinates.z = atof(elements[4].c_str());
        int slice_number = atoi(elements[5].c_str());
        int frame_number = atoi(elements[6].c_str());
		
		Event e(eventID, cell_type, coordinates, slice_number, frame_number);

		if (elements.size() > 7) {
			e.size = atof(elements[7].c_str());
			e.dilation = atof(elements[8].c_str());
			e.orientation = atof(elements[9].c_str());
			e.series = atoi(elements[10].c_str());
		}

		events.push_back(e);

    }

    fin.close();

}

// Old style annotation reading
void loadData(string filename, string &folder_name, string &series_name, int &first, int &last, int &interval, int &n_frames, string &file_format, vector<Event> &events) {
    
    /******************Read File Header*****************/
    string filename_data;
    ifstream fin;
    string text, warning_text;
    
    fin.open(filename.c_str(), ios::in);

	getline(fin, warning_text);
//
    getline(fin, text);
    folder_name = text.substr(7);
//
    getline(fin, text);
    series_name = text.substr(7);
//
    getline(fin, text);
    file_format = text.substr(13);
//    
    getline(fin, text);
    filename_data = text.substr(10);
//
    getline(fin, text);
    first = atoi(split(text, '=')[1].c_str());
//
    getline(fin, text);
    last = atoi(split(text, '=')[1].c_str());
//
    getline(fin, text);
    interval  = atoi(split(text, '=')[1].c_str());
//    interval = atoi(text.substr(9).c_str());
    getline(fin, text);
    n_frames = atoi(split(text, '=')[1].c_str());

	fin.close();

    /*************************************************/

	// read .csv annotation file
	readAnnotation(filename_data, events);
    
}

/*
*	EVENT COLLECTION FILTERING
*/

Event getWithID(vector<Event> events, int ID, string type = "Daughter1") {
	for(int i = 0; i < events.size(); i++) {
		if (events[i].cell_type==type && events[i].eventID == ID)
			return events[i];
	}
	return Event();
}

Event getDaughterWithID(vector<Event> events, int ID, string type = "Daughter1") {
	for(int i = 0; i < events.size(); i++) {
		if (events[i].cell_type==type && events[i].eventID == ID)
			return events[i];
	}
	return Event();
}

vector<Event> getEventsFromMovie(vector<Event> events, int movie_id) {
	vector<Event> found = vector<Event>();
	for(int i = 0; i < events.size(); i++) {
		if (events[i].series == movie_id) {
			found.push_back(events[i]);
		}
	}
	return found;	
}

vector<Event> findAssociatedEvents(vector<Event> events, int t, int z) {
	vector<Event> found = vector<Event>();
	for(int i = 0; i < events.size(); i++) {
		if (events[i].frame_number == t && events[i].slice_number == z) {
			found.push_back(events[i]);
		}
	}
	return found;
}

vector<Event> findAssociatedEventsOfMovie(vector<Event> events, int movie_id, int t, int z) {
	events = getEventsFromMovie(events, movie_id);
	return findAssociatedEvents(events, t, z);
}

/*
*	IMAGE LOADING
*/

void loadImage(Mat &img, string folder_name, string series_name, int z, int t, string file_format = "png") {
    stringstream filename;
    filename << folder_name << "/" << series_name << "_t" << setfill('0') << setw(3) << t;
    filename << "_z" << setfill('0') << setw(3) << z << "." << file_format;

	cout<<filename.str()<<endl;

    img = imread(filename.str());
}

void getTemplate(int n, Mat &templ, Mat &mask) {
	templ = imread((boost::format("templates/template%d.png") % n).str(), 0);
	mask = imread((boost::format("templates/mask%d.png") % n).str(), 0)/255;
}

/*
*	ANNOTATION REORGANIZING AND CREATING
*/

void separateMothersFromDaughters(string file, string file_mother, string file_daughter) {
	vector<Event> events;
	readAnnotation("E:/IDP/annotations/positive samples/training.csv", events);

	Event().writeHeaderToFileExtended(file_mother);
	Event().writeHeaderToFileExtended(file_daughter);

	for(int i = 0; i < events.size(); i++) {
		if (events[i].cell_type == "Mother")
			events[i].writeEventToFileExtended(file_mother);
		else
			events[i].writeEventToFileExtended(file_daughter);
	}
}

void extractRandom(vector<Movie> movies, int m, int d, string file_mother, string file_daughter) {
	srand (time(NULL));

	Event().writeHeaderToFileExtended(file_mother);
	Event().writeHeaderToFileExtended(file_daughter);

	string cells[] = {"Mother", "Daughter1", "Daughter2"};

	for(int i = 0; i < m+d; i++) {
		int rand_movie = i%movies.size();
		int rand_z = rand()%(movies[rand_movie].nslices - 5) + 3;
		int rand_t = rand()%movies[rand_movie].nframes + 1;
		Mat image = imread(movies[rand_movie].getPath(rand_z, rand_t));

		float resize_coeff = target_width*1.0/image.cols;
		resize(image, image, Size(image.cols*resize_coeff, image.rows*resize_coeff));

		Mat mask;
		extractCryptMask(image, mask);
		int randx, randy;
		int size = rand()%60 + 20; // between 20 and 80, calculated from positive samples
		int j = 0;
		while(true) {
			randx = rand()%(image.cols-size-20) + size/2+10;
			randy = rand()%(image.rows-size-20) + size/2+10;
			if (mask.at<uchar>(randy, randx) > 0) break;
			j++;
			if (j > 15) break;
		}
		int orientation = rand()%360;
		string cell_type;
		if (i < m) {cell_type = "Mother";} else {cell_type = "Daughter1";}

		Mat overlay(image.size(), CV_8UC3, Scalar(0,0,0));
		circle(overlay, Point(randx, randy), 3, Scalar(255, 255, 255));
		imwrite("rand/"+to_string(i)+".png", image+overlay);
		imwrite("rand/"+to_string(i)+"mask.png", mask);

		Event e = Event(i, rand_movie, cell_type,
			Point3f(randx, randy, 0), rand_z, rand_t, size, 1, orientation);

		e.resize(1/resize_coeff);
		

		if (cell_type == "Mother")
			e.writeEventToFileExtended(file_mother);
		else
			e.writeEventToFileExtended(file_daughter);
	}
}

/*
*	REFINE ANNOTATION
*/

// Find centers of mother cells just before the division
void updateMothers(string folder_name, string series_name, vector<Event> events, string extraction_path, string new_ann_path) {
	_mkdir(extraction_path.c_str());

	Event().writeHeaderToFile(new_ann_path);

	int size = 150;

	for(int i = 0; i < events.size(); i++) {
		Mat image;
		loadImage(image, folder_name, series_name, events[i].slice_number, events[i].frame_number);

		Rect patch;
		safePatchExtraction(image.size(), events[i].coordinates.x - size, events[i].coordinates.y - size,
			2*size, 2*size, patch);

		Mat overlay = Mat::zeros(image.size(), CV_8UC3);
		events[i].draw(overlay);
		imwrite((boost::format("%s/%s_id_%d_t_%d_z%d_%s.png") % extraction_path
			% series_name % events[i].eventID % events[i].frame_number
			% events[i].slice_number % events[i].cell_type
			).str(), (image+overlay)(patch));
		
		if (events[i].cell_type == "Mother") {
			Event daughter1 = getDaughterWithID(events, events[i].eventID, "Daughter1");
			int time_of_division = daughter1.frame_number;
			for (int t = events[i].frame_number+1; t <= time_of_division; t++) {
				loadImage(image, folder_name, series_name, events[i].slice_number, t);
				imwrite((boost::format("%s/%s_id_%d_t_%d_z%d_a.png") % extraction_path
					% series_name % events[i].eventID % t % events[i].slice_number).str(), (image)(patch));

			}
			Mat mother;
			loadImage(mother, folder_name, series_name, events[i].slice_number, time_of_division-1);

			Event daughter2 = getDaughterWithID(events, events[i].eventID, "Daughter2");
			Point3f center_of_division = (daughter1.coordinates + daughter2.coordinates)/2;
			int frame_size = 125;
			Rect frame;
			safePatchExtraction(mother.size(), center_of_division.x-frame_size, center_of_division.y-frame_size,
				frame_size*2, frame_size*2, frame);

			Mat cut;
			getChannel(mother(frame), 2, cut);
			resize(cut, cut, Size(cut.cols/2, cut.rows/2));

			Mat templ, mask, map;
			getTemplate(5, templ, mask);
			//templateMatching(cut, templ, mask, map);

			double minVal; double maxVal; Point minLoc; Point maxLoc;

			minMaxLoc(map, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

			//cout<<maxLoc<<endl;

			//imshow("Map", map);
			//waitKey(0);

			circle(cut, maxLoc, 2, 255, -1);

			imwrite((boost::format("%s/%s_id_%d_t_%d_z%d_y.png") % extraction_path
				% series_name % events[i].eventID % (time_of_division-1) % events[i].slice_number).str(), map*255);

			imwrite((boost::format("%s/%s_id_%d_t_%d_z%d_z.png") % extraction_path
				% series_name % events[i].eventID % (time_of_division-1) % events[i].slice_number).str(), cut);

			Event new_mother_event = Event(events[i].eventID, "Mother",
				Point3f(frame.x+maxLoc.x*2, frame.y+maxLoc.y*2, 14), events[i].slice_number, time_of_division-1);
			
			getChannel(mother, 2, mother);

			circle(mother, Point(new_mother_event.coordinates.x, new_mother_event.coordinates.y), 3, 255, -1);
			imwrite((boost::format("%s/%s_id_%d_t_%d_z%d_B.png") % extraction_path
				% series_name % events[i].eventID % (time_of_division-1) % events[i].slice_number).str(), mother);
			
			
			new_mother_event.writeEventToFile(new_ann_path);
			daughter1.writeEventToFile(new_ann_path);
			daughter2.writeEventToFile(new_ann_path);
		}
	
	}
}

// Add size, dilation, angle information to the division
void recognizeSizes(string folder_name, string series_name, vector<Event> events, string extraction_path, string new_ann_path) {
	_mkdir(extraction_path.c_str());

	Event().writeHeaderToFileExtended(new_ann_path);

	int patch_size = 40;

	Mat templ, mask;
	getTemplate(5, templ, mask);

	for(int i = 0; i < events.size(); i++) {

		//if (events[i].eventID != 5) continue;

		Mat image;
		loadImage(image, folder_name, series_name, events[i].slice_number, events[i].frame_number);
		cout<<series_name<<events[i].slice_number<<events[i].frame_number<<endl;
		float resize_coeff = 300.0/image.cols;

		resize(image, image, Size(image.cols*resize_coeff, image.rows*resize_coeff));

		//imshow("Mother", image);
		//waitKey(0);

		//circle(image, Point(events[i].coordinates.x, events[i].coordinates.y), 2,
		//	Scalar(255, 255, 255), -1);

		Mat patch;
		Rect ROI = Rect(events[i].coordinates.x*resize_coeff-patch_size,
			events[i].coordinates.y*resize_coeff-patch_size, patch_size*2, patch_size*2);
		safePatchExtractionSymmetric(image.size(), ROI);

		getChannel(image(ROI), 2, patch);

		Gerda::Range<float> size;
		Gerda::Range<float> dil;
		Gerda::Range<float> angle;

		if (events[i].cell_type == "Mother") {
			size = Gerda::Range<float>(15, 60, 5);
			dil = Gerda::Range<float>(1, 4, 0.2);
			angle = Gerda::Range<float>(0, 180, 10);
		} else {
			//double d = max((double)events[i].coordinates.z*resize_coeff*2 - 5, 10.0);
			//cout<<d<<endl;
			size = Gerda::Range<float>(15, 60, 5);
			//size = Gerda::Range<float>(d, d+10, 2);
			dil = Gerda::Range<float>(1, 5, 0.2);
			angle = Gerda::Range<float>(0, 180, 10);
		}

		int searchD = 7;
		Rect searchRegion(patch.cols/2 - searchD, patch.rows/2 - searchD,
			searchD*2, searchD*2);

		Point maxLoc;
		float max_size, max_dil, max_angle;
		vector<Mat> bins;
		templateMatching(patch, templ, mask, size, dil, angle, bins, 1, 
			maxLoc, max_size, max_dil, max_angle, searchRegion);
		//int max_bin;
		//getMaxBinAndLoc(bins, max_bin, maxLoc);
		//float best_size = size.interval*max_bin+size.min;

		events[i].size = max_size/resize_coeff;
		events[i].dilation = max_dil;
		events[i].orientation = max_angle;
		events[i].writeEventToFileExtended(new_ann_path);

		Mat best_templ;
		transform(templ, max_size, max_dil, max_angle, best_templ);

		Mat canvas = Mat::zeros(patch.size(), CV_8UC1);
		best_templ.copyTo(canvas(
			Rect(maxLoc.x - best_templ.cols/2, maxLoc.y - best_templ.rows/2, best_templ.cols, best_templ.rows)));
		Mat color[] = {patch, patch, patch + canvas};
		//color[1] = color[1] + canvas;
		//circle(patch, maxLoc, best_size/2, Scalar(255), 3);
			
		Mat merged;
		merge(color, 3, merged);
		rectangle(merged, searchRegion, Scalar(0, 255, 0), 1);
		imwrite((boost::format("%s/%s_id_%d_t_%d_z%d_%s_B.png") % extraction_path
			% series_name % events[i].eventID % events[i].frame_number % events[i].slice_number % events[i].cell_type).str(), patch);

		imwrite((boost::format("%s/%s_id_%d_t_%d_z%d_%s.png") % extraction_path
			% series_name % events[i].eventID % events[i].frame_number % events[i].slice_number % events[i].cell_type).str(), merged);


	}

}

vector<Event> reduceEvents(vector<Event> events, float radius, vector<float> prob) {
	vector<vector<int>> bags;
	for(int i = 0; i < events.size(); i++) {
		int bag = -1;
		for(int b = 0; b < bags.size(); b++) {
			if (norm(events[bags[b][0]].getPoint() - events[i].getPoint()) < radius) {
				bag = b;
				break;
			}
		}
		if (bag == -1) bags.push_back(vector<int>(1, i));
		else bags[bag].push_back(i);
	}
	vector<Event> most_prob(bags.size());
	for(int b = 0; b < bags.size(); b++) {
		float max_prob = -1; int max_i;
		for(int i = 0; i < bags[b].size(); i++) {
			if (prob[bags[b][i]] > max_prob) {
				max_prob = prob[bags[b][i]]; max_i = bags[b][i];
			}
		}
		most_prob[b] = events[max_i];
	}
	return most_prob;
}



#endif