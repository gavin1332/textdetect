#include "td/text-detect.h"

#include "test/test.h"

using namespace std;
using namespace cv;

void TextDetector::DispRects(const Mat& gray, const list<TextRect*>& trlist,
    Scalar color) {
  vector<Rect> rect_vec;
  list<TextRect*>::const_iterator itr = trlist.begin(), end = trlist.end();
  for (; itr != end; ++itr) {
    rect_vec.push_back((*itr)->ToCvRect());
  }
  TestUtils::ShowRects(gray, rect_vec, color);
}