/*
 * stroke.h
 *
 *  Created on: Sep 9, 2012
 *      Author: liuyi
 */

#ifndef TD_STROKE_H_
#define TD_STROKE_H_

#include <opencv2/core/core.hpp>

#include "core/core.h"

class Stroke {
 public:
	static const int kMaxLevel = 4;

  Stroke() {}
  virtual ~Stroke() {}

  void Extract(const cv::Mat& gray, cv::Mat* level_map);

 private:
  typedef struct context {
    cv::Mat gray;
    cv::Mat koller[2];
    cv::Mat gauss[5];
    cv::Mat dog_ext[2];
    int level;
    int half_width;

    cv::Mat dx;
    cv::Mat dy;
    cv::Mat dxx;
    cv::Mat dyy;
    cv::Mat dxy;
    cv::Mat hessian;

    cv::Point pos;
    cv::Mat dir;
    uchar pos_value;
    uchar neg_value;
    cv::Point2f shift;

    float left_resp;
    float right_resp;
  } Context;

  static const int kInitWidth = 3;

  static const uchar kFG;
  static const uchar kBG;
  static const float kShiftThres;

  void BuildScaleSpace(Context* cxt);

  void Koller(Context* cxt);

  void Refine(cv::Mat* mask);

  void CalcGrad(Context* cxt, int layer);

  void CalcDirection(Context* cxt);

  void CalcExtremalShift(Context* cxt);

  bool CalcSideResps(Context* cxt);

  float Positive(float d) {
    return (d < 0)? 0 : d;
  }
  float Negtive(float d) {
    return (d < 0)? d : 0;
  }

};

#endif /* TD_STROKE_H_ */
