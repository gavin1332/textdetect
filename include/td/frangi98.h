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
  static const double EPSILON;

  void Handle(const cv::Mat& gray, std::list<TextRect*>* trlist);
  
  void genRespMap(const cv::Mat& fgray, double sigma, cv::Mat* resp, cv::Mat* mask);
  
  void GroupRegion(const cv::Mat& binary, std::vector<Region*>* region_vec,
  		std::list<TextRect*>* trlist);
  
  Frangi98(const Frangi98&);
  void operator=(const Frangi98&);

};

#endif

