#ifndef TD_TEXTDETECTOR_H_
#define	TD_TEXTDETECTOR_H_

#include <list>

#include <opencv2/core/core.hpp>

#include "core/core.h"

class TextDetector {
public:
  TextDetector() {};
  virtual ~TextDetector() {};
  void Detect(const cv::Mat& img, std::list<TextRect*>* trlist);
protected:
  static void DispRects(const cv::Mat& gray, const std::list<TextRect*>& trlist,
      cv::Scalar color = cv::Scalar(255, 255, 255));
private:
  TextDetector(const TextDetector&);
  void operator=(const TextDetector&);

};

#endif

