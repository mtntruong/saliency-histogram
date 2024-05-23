#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <sys/time.h>
#include <cstdlib>
#include <string>

#include "getopt.h"
#include "filter.h"
#include "selector.h"
#include "state.h"

using namespace cv;
using namespace std;

typedef unsigned int uint;

const char* WINDOW  = "Particle Tracker";

const uint NUM_PARTICLES = 200;

inline void update_target_histogram(Mat& image, Rect& selection, Mat& histogram, Mat& target) {
	Mat roi(image, selection);
	roi.copyTo(target);
	Mat new_hist;
	float alpha = 0.2;

	calc_hist(roi, new_hist);
	normalize(new_hist, new_hist);

	if(histogram.empty()) {
		histogram = new_hist;
	}
	else {
		histogram = ((1.f - alpha) * histogram) + (alpha * new_hist);
		normalize(histogram, histogram);
	}
	cout << "Target updated" << endl;
}

struct StateData {
	StateData(int num_particles):
		image(),
		target(),
		target_histogram(),
		selector(WINDOW),
		selection(),
		paused(false),
		draw_particles(false),
		filter(num_particles)
	{};

	Mat image;
	Mat target;
	Mat target_histogram;
	Selector selector;
	Rect selection;
	bool paused;
	bool draw_particles;
	ParticleFilter filter;
};

State_ state_start(StateData& d) {

	if( d.selector.selecting() ) {
		cout << "state_selecting" << endl;
		return state_selecting;
	}
	else {
		return state_start;
	}
}

State_ state_selecting(StateData& d) {
	if( d.selector.valid() ) {
		cout << "state_initializing: (" << d.selection.x << ", " << d.selection.y << ", " << d.selection.width << ", " << d.selection.height  << ")" << endl;
		d.selection = d.selector.selection();
		cout << "selection: (" << d.selection.x << ", " << d.selection.y << ", " << d.selection.width << ", " << d.selection.height  << ")" << endl;
		return state_initializing(d); // Call with current frame
	}
	else {
		Mat roi(d.image, d.selector.selection());
		bitwise_not(roi, roi);
		return state_selecting;
	}
}

State_ state_initializing(StateData& d) {
	if( d.selector.selecting() ) {
		cout << "state_selecting" << endl;
		return state_selecting;
	}

	// Generate initial target histogram
	update_target_histogram(d.image, d.selection, d.target_histogram, d.target);

	// Initialize condensation filter with center of selection
	d.filter.init(d.selection);

	// Start video running if paused
	d.paused = false;

	cout << "state_tracking" << endl;
	return state_tracking(d); // Call with current frame
}

State_ state_tracking(StateData& d) {
	if( d.selector.selecting() ) {
		cout << "state_selecting" << endl;
		return state_selecting;
	}

	// Update particle filter
	d.filter.update(d.image, d.selection.size(), d.target_histogram);

	Size target_size(d.target.cols, d.target.rows);

	// Draw particles
	if( d.draw_particles )
		d.filter.draw_particles(d.image, target_size, Scalar(255, 255, 255));

	// Draw estimated state with color based on confidence
	float confidence = d.filter.confidence();

	// TODO - Make these values not arbitrary
	d.filter.draw_estimated_state(d.image, target_size,  Scalar(0, 255, 0));

	return state_tracking;
}

struct Options {
	Options()
	:num_particles(NUM_PARTICLES),
	 infile(),
	 outfile()
	{}

	int num_particles;
	string infile;
	string outfile;
};

void parse_command_line(int argc, char** argv, Options& o) {
	int c = -1;
	while( (c = getopt(argc, argv, "lo:p:")) != -1 ) {
		switch(c) {
			case 'o':
				o.outfile = optarg;
				break;
			case 'p':
				o.num_particles = atoi(optarg);
				break;
			default:
				cerr << "Usage: " << argv[0] << " [-o output_file] [-p num_particles] [input_file]" << endl << endl;
				cerr << "\t-o output_file : Optional MPEG output file" << endl;
				cerr << "\t-p num_particles: Number of particles (samples) to use, default is 200" << endl;
				cerr << "\tinput_file : Optional file to read, otherwise use camera" << endl;
				exit(1);
		}
	}

	if( optind < argc ) {
		o.infile = argv[optind];
	}

	cout << "Num particles: " << o.num_particles << endl;
	cout << "Input file: " << o.infile << endl;
	cout << "Output file: " << o.outfile << endl;

}

int main(int argc, char** argv) {
	Options o;
	parse_command_line(argc, argv, o);

	bool use_camera;
	VideoCapture cap;
	VideoWriter writer;

	// Use filename if given, else use default camera
	if( !o.infile.empty() ) {
		cap.open(o.infile);
		use_camera = false;
	}
	else {
		cap.open(0);
		use_camera = true;
	}

	if( !cap.isOpened() ) {
		cerr << "Failed to open capture device" << endl;
		exit(2);
	}

	if( !o.outfile.empty() ) {
		int fps = cap.get(CAP_PROP_FPS);
		int width = cap.get(CAP_PROP_FRAME_WIDTH);
		int height = cap.get(CAP_PROP_FRAME_HEIGHT);
		writer.open(o.outfile, VideoWriter::fourcc('j', 'p', 'e', 'g'), fps, Size(width, height));
		if( !writer.isOpened() ) {
			cerr << "Could not open '" << o.outfile << "'" << endl;
			exit(1);
		}
		use_camera = false;
	}

	// Open window and start capture
	namedWindow(WINDOW, WINDOW_FREERATIO | WINDOW_GUI_NORMAL);


	StateData d(o.num_particles);
	State state = state_start;
	Mat frame, gray;

	// Main loop

	int count = 0;
	for(;;) {

		// Start timing the loop
		timeval start_time;
		gettimeofday(&start_time, 0);

		// Capture frame
		if( !d.paused) {
			cap >> frame;
			if(frame.empty()) {
				cerr << "Error reading frame" << endl;
				break;
			}
		}
		if( use_camera ) {
			flip(frame, d.image, 1);
		}
		else {
			frame.copyTo(d.image);
		}

		if (!count) {d.paused = true;}

		// Handle keyboard input
		char c = (char)waitKey(10);
		if( c == 27 )
			break;
		switch(c) {
			case 'p':
				d.paused = !d.paused;
				break;

			case 'c':
				cout << "Tracking cancelled." << endl;
				state = state_start;
				break;

			case 'd':
				d.draw_particles = !d.draw_particles;
				cout << "Draw particles: " << d.draw_particles << endl;
				break;
		}

		// Process frame in current state
		state = state(d);

		imshow(WINDOW, d.image);
		if( writer.isOpened() and !d.paused ) {
			writer << d.image;
		}
		count++;
	}
}
