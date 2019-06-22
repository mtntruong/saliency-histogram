#include "selector.h"
#include <cmath>
using namespace cv;

void Selector::mouse_callback(int event, int x, int y, int flags, void* data) {
	Selector& self = *((Selector*)data);

	switch( event ) {
		case CV_EVENT_LBUTTONDOWN:
			self.m_selection_valid = false;
			self.m_selecting = true;
			self.m_selection = Rect(0,0,0,0);
			self.m_origin.x = x;
			self.m_origin.y = y;
			break;

		case CV_EVENT_LBUTTONUP:
			self.m_selection_valid = true;
			self.m_selecting = false;
			break;

		default:
			if( self.m_selecting ) {
				self.m_selection.x = MIN(x, self.m_origin.x);
				self.m_selection.y = MIN(y, self.m_origin.y);
				self.m_selection.width = std::abs(x - self.m_origin.x);
				self.m_selection.height = std::abs(y - self.m_origin.y);
			}
	}
}
