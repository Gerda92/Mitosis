#ifndef EVENT_H
#define	EVENT_H

#include "includes.h"
#include "Movie.h"

class Event {

public:

	Movie movie;

	int series;
	int eventID, slice_number, frame_number;
    string cell_type;
    Point3f coordinates;
	float size, dilation, orientation;

	Event() {
		this->eventID = -1;
	}

    Event(int eventID, string cell_type, Point3f coordinates, int slice_number, int frame_number) {
        this->eventID = eventID;
		this->cell_type = cell_type;
        this->coordinates = coordinates;
        this->slice_number = slice_number;
        this->frame_number = frame_number;
		this->size = 0; this->dilation = 0; this->orientation = 0;
    }

    Event(int eventID, int series, string cell_type, Point3f coordinates, int slice_number, int frame_number,
		float size, float dilation, float orientation) {
		this->series = series;
        this->eventID = eventID;
		this->cell_type = cell_type;
        this->coordinates = coordinates;
        this->slice_number = slice_number;
        this->frame_number = frame_number;
		this->size = size;
		this->dilation = dilation;
		this->orientation = orientation;
    }

	Movie getMovie() {
		if (movie.undef()) {
			movie = Movie::getMovies()[series];
		}
		return movie;
	}

	Mat loadImage() {
		Movie m = getMovie();
		return imread(m.getPath(slice_number, frame_number));
	}

	void resize(float coeff) {
        coordinates = coordinates*coeff;
		size = size*coeff;
	}
	
    Point3f getCoordinates() {
        return coordinates;
    }

    int getEventID() {
        return eventID;
    }

    void writeHeaderToFile(string filename) {
        ofstream f;
        f.open(filename.c_str(), ios::out);
        f <<"eventID,Cell Type ,Center.x,Center.y,Radius,Slice,Frame"<< endl; 
        f.close();
    }

    void writeEventToFile(string filename) {
        ofstream f;
        f.open(filename.c_str(), ios::app);
        f <<eventID << "," << cell_type << "," <<coordinates.x <<"," << coordinates.y << "," << coordinates.z <<"," <<slice_number << "," << frame_number << endl; 
        
        f.close();
    }

    void writeHeaderToFileExtended(string filename) {
        ofstream f;
        f.open(filename.c_str(), ios::out);
        f <<"eventID,Cell Type,Center.x,Center.y,Radius,Slice,Frame,Size,Dilation,Orientation,Series"<< endl; 
        f.close();
    }

    void writeEventToFileExtended(string filename) {
        ofstream f;
        f.open(filename.c_str(), ios::app);
        f <<eventID << ","  << cell_type << "," <<coordinates.x <<"," << coordinates.y << "," << coordinates.z <<"," <<slice_number << "," << frame_number <<
			"," << size << "," << dilation << "," << orientation << "," << series <<  endl; 
        
        f.close();
    }

	void draw(Mat &canvas) {
		if (cell_type == "Mother")
			cv::circle(canvas, Point(coordinates.x, coordinates.y), coordinates.z, Scalar(0, 0, 255), 3);
		else
			cv::circle(canvas, Point(coordinates.x, coordinates.y), coordinates.z, Scalar(0, 255, 255), 3);
	}

	Point getPoint() {
		return Point(coordinates.x, coordinates.y);
	}

};

#endif	/* EVENT_H */

