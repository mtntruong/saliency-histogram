#include <iostream>
#include "filter.h"

using namespace cv;
using namespace std;
using namespace saliency;

typedef unsigned int uint;

ParticleFilter::ParticleFilter(int num_particles):
		ConDensation(NUM_STATES, num_particles),m_mean_confidence(0.f){}

ParticleFilter::~ParticleFilter(){}

void ParticleFilter::init(const Rect& selection) {
	static const float DT = 1;

	// Constant velocity model with constant scale
	m_transition_matrix = (Mat_<float>(NUM_STATES, NUM_STATES) <<
			1, 0, DT,  0,  0,
			0, 1,  0, DT,  0,
			0, 0,  1,  0,  0,
			0, 0,  0,  1,  0,
			0, 0,  0,  0,  1);

	const float initial[NUM_STATES] = {selection.x + selection.width/2, selection.y + selection.height/2, 0, 0, 1.0};
	static const float std_dev[NUM_STATES] = { 2,  2,  .5,  .5,  .1};

	cout << "Init with state: [ ";
	for( uint j = 0; j < NUM_STATES; j++) {
		cout << initial[j] << " ";
	}
	cout << "]" << endl;

	init_sample_set(initial, std_dev);
}

/**
 * Update filter with measurements and time step. 
 */
Mat& ParticleFilter::update(Mat& image, const Size& target_size, Mat& target_hist) {
	Mat hist;
	Rect bounds(0,0,image.cols, image.rows);

	// Update the confidence for each particle
	uint i;
	Rect iBounds(0,0,image.cols, image.rows);
	float scale = 1.0; int width, height, x, y;
	static const float LAMBDA = 20.f;

	for( i = 0; i < m_num_particles; i++) {
		// Use this for adaptive window size
		// scale = MAX(0.1, m_particles[i](STATE_SCALE));
		m_particles[i](STATE_SCALE) = scale;
		width = round(target_size.width * scale);
		height = round(target_size.height * scale);
		x = round(m_particles[i](STATE_X)) - width / 2;
		y = round(m_particles[i](STATE_Y)) - height / 2;

		Rect region = Rect(x, y, width, height) & iBounds;
		Mat image_roi(image, region);

		// Calculate likelihood
		Mat hist;
		calc_hist(image_roi, hist);
		normalize(hist, hist);
		float bc = compareHist(target_hist, hist, CV_COMP_BHATTACHARYYA);
		float prob = 0.f;
		if(bc != 1.f) // Clamp total mismatch to 0 likelihood
			prob = exp(-LAMBDA * (bc * bc) );
		m_confidence[i] = prob;
	}

	// Project the state forward in time
	time_update();

	// Update the confidence at the mean state
	// scale = MAX(0.1, m_state(STATE_SCALE));
	m_state(STATE_SCALE) = scale;
	width = round(target_size.width * scale);
	height = round(target_size.height * scale);
	x = round(m_state(STATE_X)) - width / 2;
	y = round(m_state(STATE_Y)) - height / 2;

	Rect region = Rect(x, y, width, height) & bounds;
	Mat image_roi(image, region);

	m_mean_confidence = calc_likelihood(image_roi, target_hist);

	// Redistribute particles to re-acquire the target if the mean state moves
	// off screen.  This usually means the target has been lost due to a mismatch
	// between the modeled motion and actual motion.
	if( !bounds.contains(Point(round(m_state(STATE_X)), round(m_state(STATE_Y)))) ) {
		static const float lower_bound[NUM_STATES] = {0, 0, -.5, -.5, 1.0};
		static const float upper_bound[NUM_STATES] = {(float) image.cols, (float) image.rows, .5, .5, 2.0};

		cout << "Redistribute: " << m_state << " " << m_mean_confidence << endl;
		redistribute( lower_bound, upper_bound );
	}

	return m_state;
}

// Calculate the likelihood for a particular region
float ParticleFilter::calc_likelihood(Mat& image_roi, Mat& target_hist) {
	static const float LAMBDA = 20.f;
	static Mat hist;

	calc_hist(image_roi, hist);
	normalize(hist, hist);

	float bc = compareHist(target_hist, hist, CV_COMP_BHATTACHARYYA);
	float prob = 0.f;
	if(bc != 1.f) // Clamp total mismatch to 0 likelihood
		prob = exp(-LAMBDA * (bc * bc) );
	return prob;
}

void ParticleFilter::draw_estimated_state(Mat& image, const Size& target_size, const Scalar& color) {
	Rect bounds(0,0, image.cols, image.rows);

	int width = round(target_size.width * m_state(STATE_SCALE));
	int height = round(target_size.height * m_state(STATE_SCALE));
	int x = round(m_state(STATE_X)) - width/2;
	int y = round(m_state(STATE_Y)) - height/2;
	Rect rect = Rect(x, y, width, height) & bounds;
	rectangle(image, rect, color, 2);
	circle(image, Point(x + width/2,y + height/2), 5, Scalar(0,255,0), CV_FILLED);
	cout << "Target center: (" << x + width/2 << ", " << y + height/2 << ")"<< endl;
}

void ParticleFilter::draw_particles(Mat& image, const Size& target_size, const Scalar& color) {
	Rect bounds(0,0, image.cols, image.rows);

	for(uint i = 0; i < m_num_particles; i++) {
		int width = round(target_size.width * m_particles[i](STATE_SCALE));
		int height = round(target_size.height * m_particles[i](STATE_SCALE));
		int x = round(m_particles[i](STATE_X)) - width/2;
		int y = round(m_particles[i](STATE_Y)) - height/2;
		Rect rect = Rect(x, y, width, height) & bounds;
		rectangle(image, rect, color, 1);
	}
}

void ParticleFilter::redistribute(const float lbound[], const float ubound[]) {
	for( uint i = 0; i < m_num_particles; i++ )	{
		for( uint j = 0; j < m_num_states; j++ ) {
			float r = m_rng.uniform(lbound[j], ubound[j]);
			m_particles[i](j) = r;
		}
		m_confidence[i] = 1.0 / (float)m_num_particles;
	}
}

// Saliency-based weighted color histogram
void calc_hist(Mat& bgr, Mat& hist) {
	static const int channels[] = {0, 1, 2};
	static const int hist_size[] = {8, 8, 8};
	static const float pixel_range[] = {0, 255};
	static const float* ranges[] = {pixel_range, pixel_range, pixel_range};
	static const Mat mask;
	static const int dims = 3;

	Mat srcs[] = {bgr};

	// Create saliency map
	Ptr<Saliency> saliencyAlgorithm = Saliency::create( "SPECTRAL_RESIDUAL" );
	Mat image(bgr);
	Mat saliencyMap, printMap;
	double optimalThresh1, optimalThresh2;

	// Calculate two-level Otsu thresholds
	if( saliencyAlgorithm->computeSaliency( image, saliencyMap ) ) {

		saliencyMap.convertTo(printMap, CV_8UC1, 255, 0);

		Mat hist;
		int histSize = 256;
		int N = printMap.cols * printMap.rows;
		float range[] = { 0, 255 };
		const float *ranges[] = { range };
		calcHist( &printMap, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false );

		double W0K, W1K, W2K, M0, M1, M2, currVarB, maxBetweenVar, M0K, M1K, M2K, MT;

		optimalThresh1 = 0;	optimalThresh2 = 0;
		W0K = 0; W1K = 0;
		M0K = 0; M1K = 0;
		MT = 0;
		maxBetweenVar = 0;

		for (int k = 0; k <= 255; k++)
			MT += k * ((double)hist.at<float>(k) / (double) N);

		for (int t1 = 0; t1 <= 255; t1++) {
			W0K += (double)hist.at<float>(t1) / (double) N;
			M0K += t1 * ((double)hist.at<float>(t1) / (double) N);
			M0 = M0K / W0K;
			W1K = 0; M1K = 0;

			for (int t2 = t1 + 1; t2 <= 255; t2++) {
				W1K += (double)hist.at<float>(t2) / (double) N;
				M1K += t2 * ((double)hist.at<float>(t2) / (double) N);

				M1 = M1K / W1K;
				W2K = 1 - (W0K + W1K);
				M2K = MT - (M0K + M1K);

				if (W2K <= 0) break;

				M2 = M2K / W2K;
				currVarB = W0K * (M0 - MT) * (M0 - MT) +
						W1K * (M1 - MT) * (M1 - MT) +
						W2K * (M2 - MT) * (M2 - MT);

				if (maxBetweenVar < currVarB) {
					maxBetweenVar = currVarB;
					optimalThresh1 = t1; optimalThresh2 = t2;
				}
			}
		}
	}

	// Create masks
	Mat binMap1, binMap2;
	threshold(printMap, binMap1, optimalThresh1, 255, THRESH_BINARY);
	threshold(printMap, binMap2, optimalThresh2, 255, THRESH_BINARY);

	// Calculate histograms
	Mat histT0, histT1, histT2;
	calcHist(srcs, sizeof(srcs), channels, binMap1, histT1, dims, hist_size, ranges, true, false);
	calcHist(srcs, sizeof(srcs), channels, binMap2, histT2, dims, hist_size, ranges, true, false);
	calcHist(srcs, sizeof(srcs), channels, mask   , histT0, dims, hist_size, ranges, true, false);
	hist = histT0 + histT1 + histT2;
}
