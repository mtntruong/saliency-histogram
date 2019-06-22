#include <opencv2/opencv.hpp>
#include <opencv2/saliency.hpp>
#include "condens.h"

void calc_hist(cv::Mat& bgr, cv::Mat& hist);

class ParticleFilter : private ConDensation {
	public:

		enum FilterStates {
			STATE_X,
			STATE_Y,
			STATE_X_VEL,
			STATE_Y_VEL,
			STATE_SCALE,
			NUM_STATES
		};

		ParticleFilter(int num_particles);
		virtual ~ParticleFilter();

		void init(const cv::Rect& selection);

		cv::Mat& update(cv::Mat& image, const cv::Size& target_size, cv::Mat& target_hist);

		void draw_estimated_state(cv::Mat& image, const cv::Size& target_size, const cv::Scalar& color);

		void draw_particles(cv::Mat& image, const cv::Size& target_size, const cv::Scalar& color);

		void redistribute(const float lower_bound[], const float upper_bound[]);

		const cv::Mat& state() const { return m_state; }

		float confidence() const { return m_mean_confidence; };

	private:

		float calc_likelihood(cv::Mat& image_roi, cv::Mat& target_hist);

		float m_mean_confidence;

};
