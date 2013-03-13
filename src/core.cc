/*
 * core.cc
 *
 *  Created on: Sep 5, 2012
 *      Author: liuyi
 */

#include "core/core.h"

#include <stack>
#include <list>
#include <fstream>

#include <opencv2/imgproc/imgproc.hpp>

#include "td/stroke.h"
#include "utils/common/common.h"
#include "utils/math/math.h"
#include "utils/test/test.h"

using namespace std;
using namespace cv;

void ConnComp::CalcProperties(const Mat& gray) {
	int size = pix_vec_.size();
	area_ratio_ = (float) size / (Width() * Height());
	aspect_ratio_ =	(float) Width() / Height();

	uchar pix_value[size];
	for (int i = 0; i < size; ++i) {
		pix_value[i] = gray.at<uchar>(pix_vec_[i]->pos());
	}
	CalcMedianGray(pix_value, size);
	CalcGrayStdDev(pix_value, size);
	CalcMaskProperties();
}

void ConnComp::CalcMaskProperties() {
	Mat mask;
	BuildBorderedMask(&mask);

	perimeter_ = 0;
	memset(hcross, 0, kCrossVarNum * sizeof(int));
	memset(vcross, 0, kCrossVarNum * sizeof(int));

	int ori_width = Width();
	int ori_height = Height();
	int step = mask.step1();
	for (int y = 0; y < ori_height; ++y) {
		uchar* ptr = mask.ptr(y + 1) + 1;
		for (int x = 0; x < ori_width; ++x, ++ptr) {
			if (*ptr == kFG) {
				if (*(ptr - step - 1) == kBG || *(ptr - step) == kBG
						|| *(ptr - step + 1) == kBG || *(ptr - 1) == kBG
						|| *(ptr + 1) == kBG || *(ptr + step - 1) == kBG
						|| *(ptr + step) == kBG || *(ptr + step + 1) == kBG) {
					++perimeter_;
				}

				if (*(ptr - 1) == kBG) {
					if (y == ori_height / 4) {
						++hcross[0];
					}
					if (y == ori_height / 2) {
						++hcross[1];
					}
					if (y == ori_height * 3 / 4) {
						++hcross[2];
					}
				}

				if (*(ptr - step) == kBG) {
					if (x == ori_width / 4) {
						++vcross[0];
					}
					if (x == ori_width / 2) {
						++vcross[1];
					}
					if (x == ori_width * 3 / 4) {
						++vcross[2];
					}
				}
			}
		}
	}

	if (TotalHCross() < 4 && aspect_ratio_ < 0.3) {
		is_upright_bar_ = true;
	}
}

void ConnComp::BuildBorderedMask(Mat* mask) {
	Rect rect = ToCvRect();
	*mask = Mat::zeros(Size(rect.width + 2, rect.height + 2), CV_8UC1);
	Point shift(1-rect.x, 1-rect.y);
	vector<Pixel*>::const_iterator it = pix_vec_.begin(), end = pix_vec_.end();
	for (; it != end; ++it) {
		mask->at<uchar>((*it)->pos() + shift) = kFG;
	}
}

void ConnComp::CalcGrayStdDev(uchar* pix_value, int size) {
	float mean = 0;
	for (int i = 0; i < size; ++i) {
		mean += pix_value[i];
	}
	mean /= size;

	for (int i = 0; i < size; ++i) {
		gray_std_dev_ += (pix_value[i] - mean) * (pix_value[i] - mean);
	}
	gray_std_dev_ = sqrt(gray_std_dev_ / size);
}

// 条件要宽松
void ConnComp::CheckValidation() {
	if (!is_valid_) return;

	int max_h_thres = (level_ == Stroke::kMaxLevel)? 40 : 200 * (Stroke::kMaxLevel - level_); //300
	if (Height() < 10 || Height() > max_h_thres) Invalidate();

	if (aspect_ratio_ > 1.5f || aspect_ratio_ < 0.1f) Invalidate();
//	if (aspect_ratio_ > 2.5f || aspect_ratio_ < 0.1f) Invalidate();

	if (area_ratio_ < 0.2f || area_ratio_ > 0.65f) Invalidate();

	for (int i = 0; i < kCrossVarNum; ++i) {
		if (hcross[i] > 5 || vcross[i] > 5) Invalidate();
	}

//	float child_area_ratio = (float) SumChildrenArea() / pix_vec_.size();
//	if (children_.size() > 4 && child_area_ratio > 0.5f) Invalidate2Root();

	if (TotalVCross() > 20) Invalidate2Leaf();
}

int ConnComp::SumChildrenArea() {
	if (children_area_ != -1) return children_area_;

	children_area_ = 0;
	if (!IsLeaf()) {
		RegionItr it = children_.begin(), end = children_.end();
		for (; it != end; ++it) {
			ConnComp* cc = static_cast<ConnComp*>(*it);
			children_area_ += cc->pix_vec_.size();
		}
	}
	return children_area_;
}

void ConnComp::Invalidate2Root() {
	ConnComp* cc = this;
	while (cc->parent()) {
		cc->Invalidate();
		cc = static_cast<ConnComp*>(cc->parent());
	}
}

void ConnComp::Invalidate2Leaf() {
	stack<ConnComp*> cc_stack;
	cc_stack.push(this);
	while (!cc_stack.empty()) {
		ConnComp* cc = cc_stack.top();
		cc_stack.pop();
		cc->Invalidate();
		cc->is_validate2leaf_ = true;
		if (cc->IsLeaf()) continue;

		RegionItr it = cc->children_.begin(), end = cc->children_.end();
		for (; it != end; ++it) {
			ConnComp* temp = static_cast<ConnComp*>(*it);
			if (!temp->is_validate2leaf_) {
				cc_stack.push(static_cast<ConnComp*>(temp));
			}
		}
	}
}


TextRect::TextRect(const list<ConnComp*>::const_iterator begin,
    const list<ConnComp*>::const_iterator end) {
	new (this) TextRect(INT_MAX, INT_MAX, 0, 0);

  list<ConnComp*>::const_iterator it = begin;
  for (; it != end; ++it) {
    AddConnComp(*it);
  }
}

void TextRect::CheckWidthContainedOfAheadRegions(list<ConnComp*>::iterator fit) {
	ConnComp* curr = *fit;
	list<ConnComp*>::reverse_iterator rit(fit), rend = cc_list_.rend();
	for (; rit != rend && (*rit)->x1() >= curr->x1(); ++rit) {
		(*rit)->set_width_contained();
	}
}

list<ConnComp*>::iterator TextRect::FindInsertPos(list<ConnComp*>::iterator fit) {
	ConnComp* curr = *fit;
	list<ConnComp*>::reverse_iterator rit(fit), rend = cc_list_.rend();
	for (; rit != rend && (*rit)->width_contained() && (*rit)->x2() > curr->x2();
			++rit) {}
	return rit.base();
}

void TextRect::Insert2List(ConnComp* r) {
	if (cc_list_.empty() || FollowListTail(r)) {
		cc_list_.push_back(r);
	} else {
		list<ConnComp*>::iterator begin = cc_list_.begin(), end = cc_list_.end();
		list<ConnComp*>::iterator it = begin;
		while (it != end && ((*it)->width_contained() || r->x2() > (*it)->x2())) {
			++it;
		}

		if (it == end) {
			it = cc_list_.insert(it, r);
			CheckWidthContainedOfAheadRegions(it);
		} else if (r->x2() < (*it)->x2()) {
			if (r->x1() >= (*it)->x1()) {
				r->set_width_contained();
				cc_list_.insert(it, r);
			} else {
				it = FindInsertPos(it);
				it = cc_list_.insert(it, r);
				CheckWidthContainedOfAheadRegions(it);
			}
		} else {
			if (r->x1() >= (*it)->x1()) {
				r->set_width_contained();
				cc_list_.insert(it, r);
			} else {
				it = cc_list_.insert(++it, r);
				CheckWidthContainedOfAheadRegions(it);
			}
		}
	}
}

void TextRect::AddConnComp(ConnComp* r) {
  Insert2List(r);

  median_gray_mean_ = static_cast<uchar>((median_gray_mean_*RegionNum()
  	+ r->median_gray())/(RegionNum() + 1));

  if (r->Height() > max_cc_h_) {
    max_cc_h_ = r->Height();
  }

  x1_ = min(x1_, r->x1());
  y1_ = min(y1_, r->y1());
  x2_ = max(x2_, r->x2());
  y2_ = max(y2_, r->y2());
}

int TextRect::CharNum() const {
	int count = 0;
	list<ConnComp*>::const_iterator it = cc_list_.begin(), end = cc_list_.end();
	for (; it != end; ++it) {
		if (!(*it)->width_contained()) ++count;
	}
	return count;
}

int TextRect::MedianWidth() const {
  vector<int> wvec;
  list<ConnComp*>::const_iterator rit = cc_list_.begin();
  for (; rit != cc_list_.end(); ++rit) {
    if (!(*rit)->width_contained()) wvec.push_back((*rit)->Width());
  }

  size_t n = wvec.size() / 2;
  nth_element(wvec.begin(), wvec.begin() + n, wvec.end());
  return wvec[n];
}

int TextRect::MedianHeight() const {
  vector<int> hvec;
  list<ConnComp*>::const_iterator rit = cc_list_.begin();
  for (; rit != cc_list_.end(); ++rit) {
    if ((*rit)->width_contained()) {
      int top = INT_MAX;
      int bottom = 0;
      for (; (*rit)->width_contained(); ++rit) {
        top = min(top, (*rit)->y1());
        bottom = max(bottom, (*rit)->y2());
      }
      top = min(top, (*rit)->y1());
      bottom = max(bottom, (*rit)->y2());

      hvec.push_back(bottom - top + 1);
    } else {
      hvec.push_back((*rit)->Height());
    }
  }

  size_t n = (hvec.size() - 1) / 2;
  nth_element(hvec.begin(), hvec.begin() + n, hvec.end());
  return hvec[n];
}

int TextRect::MedianInterval() const {
  vector<int> interval_vec;
  list<ConnComp*>::const_iterator end = cc_list_.end();
  list<ConnComp*>::const_iterator ait = cc_list_.begin();
  for (; (*ait)->width_contained(); ++ait) {}

  list<ConnComp*>::const_iterator bit = ait;
  for (; ait != end; ait = bit) {
    for (++bit; bit != end && (*bit)->width_contained(); ++bit) {}
    if (bit == end) break;

    int interval = (*bit)->x1() - (*ait)->x2();
    interval_vec.push_back(interval);
  }

  size_t n = (interval_vec.size() - 1) / 2;
  nth_element(interval_vec.begin(), interval_vec.begin() + n, interval_vec.end());
  return interval_vec[n];
}

void TextRect::MedianY12(int* my1, int* my2) const {
  vector<int> y1vec;
  vector<int> y2vec;
  list<ConnComp*>::const_iterator it = cc_list_.begin(), end = cc_list_.end();
  for (; it != end; ++it) {
    if ((*it)->width_contained()) {
      int top = INT_MAX;
      int bottom = 0;
      for (; (*it)->width_contained(); ++it) {
        top = min(top, (*it)->y1());
        bottom = max(bottom, (*it)->y2());
      }
      top = min(top, (*it)->y1());
      bottom = max(bottom, (*it)->y2());

      y1vec.push_back(top);
      y2vec.push_back(bottom);
    } else {
      y1vec.push_back((*it)->y1());
      y2vec.push_back((*it)->y2());
    }
  }

  size_t n = (y1vec.size() - 1) / 2;
  nth_element(y1vec.begin(), y1vec.begin() + n, y1vec.end());
  *my1 = y1vec[n];
  nth_element(y2vec.begin(), y2vec.begin() + n, y2vec.end());
  *my2 = y2vec[n];
}

TextRect TextRect::FrontChar() const {
  list<ConnComp*>::const_iterator begin = cc_list_.begin();
  list<ConnComp*>::const_iterator rit = begin;
  for (; (*rit)->width_contained(); ++rit) {}
  return TextRect(begin, ++rit);
}

TextRect TextRect::BackChar() const {
  list<ConnComp*>::const_iterator end = cc_list_.end();
  list<ConnComp*>::const_iterator rit = end;
  for (--(--rit); (*rit)->width_contained(); --rit) {}
  return TextRect(++rit, end);
}

