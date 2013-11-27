#ifndef TD_FRANGI98_H_
#define	TD_FRANGI98_H_

#include <list>

#include <opencv2/core/core.hpp>

#include "text-detect.h"

class Frangi98 : public TextDetector {
public:
  Frangi98() {}
  virtual ~Frangi98() {}
  
	void Detect(const cv::Mat& img, std::list<TextRect*>* trlist);
  
private:
  typedef struct context {
    cv::Mat img;
    cv::Mat gray;
    Polarity polarity;
  } Context;
  
  void DoSomething(const cv::Mat& gray, cv::Mat* output);
  
  void HandleOnePolarity(Context* cxt, std::list<TextRect*>* trlist);
  
  Frangi98(const Frangi98&);
  void operator=(const Frangi98&);

};

#endif

