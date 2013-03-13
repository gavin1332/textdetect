/*
 * core.h
 *
 *  Created on: Sep 5, 2012
 *      Author: liuyi
 */

#ifndef CORE_CORE_H_
#define CORE_CORE_H_

#include <string>
#include <list>
#include <cmath>
#include <sstream>

#include <opencv2/core/core.hpp>

#include "utils/svm/interface.h"
#include "utils/image/mser.h"

enum Polarity { POS = 0, NEG = 1 };

class ConnComp : public Region {
 public:
	ConnComp(int level) : Region(level), is_valid_(true), area_ratio_(0),
			aspect_ratio_(0), is_upright_bar_(false), perimeter_(0), gray_std_dev_(FLT_MAX),
			median_gray_(127), width_contained_(false), is_validate2leaf_(false),
			children_area_(-1) {
		memset(hcross, 0, kCrossVarNum * sizeof(int));
	}
	~ConnComp() {}

	bool is_valid() const { return is_valid_; }
	void Invalidate() { is_valid_ = false; }

  bool width_contained() const { return width_contained_; }
  void set_width_contained() { width_contained_ = true; }

  uchar median_gray() const { return median_gray_; }
  int gray_std_dev() const { return gray_std_dev_; }

  float area_ratio() const { return area_ratio_; }
  float aspect_ratio() const { return aspect_ratio_; }

  bool is_upright_bar() const { return is_upright_bar_; }

	float HRatio(const ConnComp* cc) const {
    return (float) std::min(Height(), cc->Height()) / std::max(Height(), cc->Height());
  }

	float WRatio(const ConnComp* cc) const {
    return (float) std::min(Width(), cc->Width()) / std::max(Width(), cc->Width());
  }

  bool MatchY(const ConnComp* cc) const {
    return (abs(y1_ - cc->y1()) < Height() / 6 && abs(y2_ - cc->y2()) < Height() / 6)
        || (abs(y1_ - cc->y1()) < Height() / 4 && HRatio(cc) > 0.85f);
  }

	void CalcProperties(const cv::Mat& gray);
	void CalcMaskProperties();

	void CheckValidation();

	int TotalHCross() const { return hcross[0] + hcross[1] + hcross[2]; }

	int TotalVCross() const { return vcross[0] + vcross[1] + vcross[2]; }

	bool IsMoreInformativeThan(ConnComp* other) {
		int infor = TotalHCross() + TotalVCross();
		int other_infor = other->TotalHCross() + other->TotalVCross();
		if (infor > other_infor) return true;
		if (infor == other_infor && aspect_ratio_ < other->aspect_ratio_) return true;
		return false;
	}

	int SumChildrenArea();

 private:
	typedef std::vector<Region*>::iterator RegionItr;

	static const uchar kFG = 255;
	static const uchar kBG = 0;

	static const int kCrossVarNum = 3;

	bool is_valid_;

	float area_ratio_;
	float aspect_ratio_;
	bool is_upright_bar_;
	int perimeter_;
	int hcross[kCrossVarNum];
	int vcross[kCrossVarNum];
	float gray_std_dev_;
	uchar median_gray_;

  // Is its width contained by another wider region.
  bool width_contained_;

  bool is_validate2leaf_;

  int children_area_;

  void Invalidate2Root();

  void Invalidate2Leaf();

	void CalcMedianGray(uchar* pix_value, int size) {
		int n = (size - 1) / 2;
		std::nth_element(pix_value, pix_value + n, pix_value + size);
		median_gray_ = pix_value[n];
	}
	void CalcGrayStdDev(uchar* pix_value, int size);

	void BuildBorderedMask(cv::Mat* map);

  ConnComp();

};


class TextRect {
 public:
  TextRect(int x1, int y1, int x2, int y2, const std::string& text = "") :
  		x1_(x1), y1_(y1), x2_(x2), y2_(y2), text_(text), max_cc_h_(0),
  		median_gray_mean_(0) {}

  TextRect(ConnComp* r) {
  	new (this) TextRect(INT_MAX, INT_MAX, 0, 0);
  	AddConnComp(r);
  }

  TextRect(const std::list<ConnComp*>::const_iterator begin,
      const std::list<ConnComp*>::const_iterator end);

  int x1() const { return x1_; }
  int y1() const { return y1_; }
  int x2() const { return x2_; }
  int y2() const { return y2_; }
  const std::string& text() { return text_; }
  void set_text(const std::string& text) { text_ = text; }

  int Width() const { return x2_ - x1_ + 1; }
  int Height() const { return y2_ - y1_ + 1; }
  int Area() const { return Width() * Height(); }
  cv::Rect ToCvRect() const { return cv::Rect(x1_, y1_, Width(), Height()); }

  std::string ToString() {
    std::stringstream ss;
    ss << "[" << x1_ << " " << y1_ << " " << x2_ << " " << y2_ << "]";
    return ss.str();
  }

  int max_cc_h() const { return max_cc_h_; }

  uchar median_gray_mean() { return median_gray_mean_; }

  int CharNum() const;

  int RegionNum() const { return cc_list_.size(); }

  const std::list<ConnComp*>& cc_list() const { return cc_list_; }

  int MedianWidth() const;

  int MedianHeight() const;

  int MedianInterval() const;

  void MedianY12(int* my1, int* my2) const;

  int DistHorizon(const ConnComp* r) const {
    if (r->x2() < x1_ || r->x1() > x2_) {
      return std::min(abs(x1_ - r->x2()), abs(x2_ - r->x1()));
    } else {
      return 0;
    }
  }

  bool Contains(const ConnComp* r) const {
    return r->x1() >= x1_ && r->y1() >= y1_
        && r->x2() <= x2_ && r->y2() <= y2_;
  }

  const ConnComp* NeighborRegion(const ConnComp* r) const {
    return (r->x1() > x1_ + Width() / 2)? BackRegion() : FrontRegion();
  }

  TextRect FrontChar() const;
  TextRect BackChar() const;

  void AddConnComp(ConnComp* r);

  // TODO
  void TuneRect(float factor) {
  	x1_ *= factor;
  	y1_ *= factor;
  	x2_ *= factor;
  	y2_ *= factor;
  }

 private:
  int x1_;
  int y1_;
  int x2_;
  int y2_;
  std::string text_;

  int max_cc_h_;
  uchar median_gray_mean_;

  std::list<ConnComp*> cc_list_;

  void Insert2List(ConnComp* r);

  bool FollowListTail(ConnComp* r) {
  	return (r->x1() > cc_list_.back()->x1() && r->x2() > cc_list_.back()->x2());
  }

  void CheckWidthContainedOfAheadRegions(std::list<ConnComp*>::iterator fit);
  std::list<ConnComp*>::iterator FindInsertPos(std::list<ConnComp*>::iterator fit);

  const ConnComp* FrontRegion() const { return cc_list_.front(); }
  const ConnComp* BackRegion() const { return cc_list_.back(); }

};

#endif /* CORE_CORE_H_ */
