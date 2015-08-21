#include "includes.h"

#ifndef MOVIE_H
#define MOVIE_H

class Movie {

	static vector<Movie> Movie::movies;

public:

	string folder_name, series_name, file_format;
	int id, nslices, nframes;
	Size resolution;

	Movie() {};

	Movie(int idm, string folder, string series, string fmt, int z, int t) :
		id(idm), folder_name(folder), series_name(series), file_format(fmt), nslices(z), nframes(t) {};

	bool undef() {
		return folder_name == "";
	}

	string getName(int z, int t) {
		stringstream filename;
		filename << series_name << "_t" << setfill('0') << setw(3) << t;
		filename << "_z" << setfill('0') << setw(3) << z << "." << file_format;
		return filename.str();
	}

	string getPath(int z, int t) {
		stringstream filename;
		filename << folder_name << "/" << getName(z, t);
		return filename.str();
	}

	Mat loadImage(int z, int t) {
		return imread(getPath(z, t));
	}

	// management of movie list

	static vector<Movie> getMovies();

	static int findSeries(string series);

};

#endif