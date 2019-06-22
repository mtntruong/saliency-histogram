#include <opencv2/opencv.hpp>

class Selector {
	public:
		Selector(const char* window)
			:m_selection_valid(false),
			 m_selecting(false),
			 m_selection(),
			 m_origin()
		{ cv::setMouseCallback(window, mouse_callback, this); }

		bool valid() const { return m_selection_valid; }

		bool selecting() const { return m_selecting; }

		const cv::Rect& selection() const { return m_selection; }

	private:
		static void mouse_callback(int event, int x, int y, int flags, void* data);

	private:
		bool m_selection_valid;
		bool m_selecting;
		cv::Rect m_selection;
		cv::Point m_origin;
};
