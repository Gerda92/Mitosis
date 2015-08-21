#include "Movie.h"

vector<Movie> Movie::movies = vector<Movie>();

vector<Movie> Movie::getMovies() {

	if (movies.size() > 0) return movies;

	string filename = "E:/IDP/movies.txt";

	string filename_data;
	ifstream fin;
	string text;
    
	fin.open(filename.c_str(), ios::in);

	while (std::getline(fin, text)) {

		if (text == "") break;

		getline(fin, text);
		int id = atoi(text.c_str());

		std::getline(fin, text);
		string folder_name = text.substr(7);
	//
		getline(fin, text);
		string series_name = text.substr(7);
	//
		getline(fin, text);
		string file_format = text.substr(13);
	//
		getline(fin, text);
		int nslices = atoi(split(text, '=')[1].c_str());
	//
		getline(fin, text);
		int nframes  = atoi(split(text, '=')[1].c_str());
		
		Movie m(id, folder_name, series_name, file_format, nslices, nframes);

		// add resolution
		Mat image = imread(m.getPath(1, 1));
		m.resolution = image.size();

		movies.push_back(m);

	}

	return movies;
}

int Movie::findSeries(string series) {
	if (movies.size() == 0) getMovies();
	for(int i = 0; i < movies.size(); i++) {
		if (movies[i].series_name == series) return i;
	}
	return -1;
}