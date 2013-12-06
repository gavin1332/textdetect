// Copyright (c) 2010-2011, Tuji
// All rights reserved.
// 
// ${license}
//
// Author: LIU Yi

#ifndef TD_TEXT_DETECT_H_
#define TD_TEXT_DETECT_H_

#include <string>
#include <vector>
#include <list>

#include <opencv2/core/core.hpp>

#include "core/core.h"
#include "td/text-detect.h"
#include "math/math.h"

class StrokeFilter : public TextDetector {
 public:
  StrokeFilter() {}
  virtual ~StrokeFilter() {}
  
	void Detect(const cv::Mat& img, std::list<TextRect*>* trlist);

 private:
  typedef std::vector<Region*>::iterator RegionItr;

  typedef struct context {
    cv::Mat img;
    cv::Mat gray;
    cv::Mat level_map[2];
//    cv::Mat koller[2];
    Polarity polarity;

    cv::Mat* CurrLevelMap() { return level_map + polarity; }
  } Context;

  static const uchar kFG;
  static const uchar kBG;
  static const uchar kInvalid;
  static const uchar kMark;

  static const float kFloatEps;

  static void HandleOnePolarity(Context* cxt, std::list<TextRect*>* trlist);

  static void RemoveNoise(Context* cxt);

	static bool CompareY1(Region* r1, Region* r2) {
	 return r1->y1() < r2->y1();
	}

	static void CheckCCValidation(Context* cxt, std::vector<Region*>* region_pool);

  // TODO cxt ONLY FOR SHOW IMAGE
  static void CheckCCRelation(Context* cxt, std::vector<Region*>* region_pool);

  static void CheckParentCC(Context* cxt, std::vector<Region*>* region_pool);

  static void CheckAncestorCC(Context* cxt, std::vector<Region*>* region_pool);

  static void GroupRegion(Context* cxt, std::vector<Region*>* region_vec,
  		std::list<TextRect*>* trlist);

  static void Refine(Context* cxt, std::list<TextRect*>* trlist);

  static bool CheckHorizon(const TextRect* tr, const ConnComp* cc);

  static bool IsInSearchScope(const TextRect* tr, const ConnComp* r) {
  	return r->y1() < tr->y2() - tr->max_cc_h() / 4;
  }

  static bool IsSimilarWidth(ConnComp* c1, ConnComp* c2) {
  	return c1->WRatio(c2) > 0.75f;
  }

  static bool IsSimilarHeight(ConnComp* c1, ConnComp* c2) {
  	return c1->HRatio(c2) > 0.6f;
  }

  static bool IsSimilarStdDev(ConnComp* c1, ConnComp* c2) {
  	return abs(c1->gray_std_dev() - c2->gray_std_dev()) < 10;
  }

  static void OverlapAnalyse(std::list<TextRect*>* trlist);

  static void Split(std::list<TextRect*>* trlist);

  static void ShowValidCCs(Context* cxt, RegionItr begin, RegionItr end);

  static void BuildMapWithValidCCs(Context* cxt, cv::Mat* map,
  		RegionItr begin, RegionItr end);

  static void SaveRegionTree(const std::string& dir_path, Region* root);

//  void MarkCC(std::list<Region>* cclist);
//
//  static int Surrounding(const cv::Mat& stroke, cv::Point pos,
//      std::vector<cv::Point>* shift_vec = NULL);
//

//
//  static void TuneRect(std::list<TextRect>* trlist);
//
//  static void RestoreSize(std::list<TextRect>* trlist, float scale);

};

#endif  // TD_TEXT_DETECT_H_
