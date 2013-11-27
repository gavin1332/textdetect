#include "td/frangi98.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "core/core.h"
#include "utils/test/test.h"

using namespace std;
using namespace cv;

void Frangi98::Detect(const cv::Mat& img, std::list<TextRect*>* trlist) {
	extern bool SHOW_FINAL_;

  Context cxt;
	cxt.img = img;

	cvtColor(img, cxt.gray, CV_BGR2GRAY);
  
  Mat output;
  DoSomething(cxt.gray, &output);

//	cxt.polarity = POS;
//	HandleOnePolarity(&cxt, trlist);
//	cxt.polarity = NEG;
//	HandleOnePolarity(&cxt, trlist);
//
//  if (SHOW_FINAL_) DispRects(cxt.img, *trlist, Scalar(255, 255, 255));
}

void Frangi98::HandleOnePolarity(Context* cxt, std::list<TextRect*>* trlist) {
  
}

void Frangi98::DoSomething(const Mat& gray, Mat* output) {
  const int width = gray.cols;
  const int height = gray.rows;
  
  Mat fgray;
  gray.convertTo(fgray, CV_64FC1);
  
  // TODO: check proper initial parameters
  const int init_w = 3;
  const double k_sigma = sqrt(2);
  float sigma = init_w / (2 * sqrt(3));
  float delta_sigma = sigma;
  Mat gauss = fgray;
  const int N = 2;
  const double beta = 0.5;
  // TODO: tune
  const double gamma = 20;
  double lambda[2];
  Mat dxx, dyy, dxy;
  Mat hessian(2, 2, CV_64FC1);
  Mat inner(gray.size(), CV_64FC1, 0.0);
  Mat result(gray.size(), CV_64FC1, 0.0);
  Mat mask8U(gray.size(), CV_8UC1);
  Mat mask64F(gray.size(), CV_64FC1);
  double max_S_checker = 0;
  for (int i = 0; i < N; ++i) {
    if (delta_sigma < 1.0) {
      GaussianBlur(fgray, gauss, Size(0, 0), sigma);
    } else {
      GaussianBlur(gauss, gauss, Size(0, 0), delta_sigma);
    }
    // TODO: tune ksize
    Sobel(gauss, dxx, CV_64F, 2, 0, 1, 1/*TODO normalize*/, 0, BORDER_REPLICATE);
    Sobel(gauss, dyy, CV_64F, 0, 2, 1, 1/*TODO normalize*/, 0, BORDER_REPLICATE);
    Sobel(gauss, dxy, CV_64F, 1, 1, 3, 1/*TODO normalize*/, 0, BORDER_REPLICATE);
    for (int y = 0; y < height; ++y) {
      double* dxx_ptr = dxx.ptr<double>(y);
      double* dyy_ptr = dyy.ptr<double>(y);
      double* dxy_ptr = dxy.ptr<double>(y);
      double* inner_ptr = inner.ptr<double>(y);
      for (int x = 0; x < width; ++x) {
        hessian.at<double>(0, 0) = dxx_ptr[x];
        hessian.at<double>(0, 1) = dxy_ptr[x];
        hessian.at<double>(1, 0) = dxy_ptr[x];
        hessian.at<double>(1, 1) = dyy_ptr[x];
        hessian *= sigma * sigma;

        Mat eival, eivec;
        eigen(hessian, eival, eivec);
        lambda[0] = eival.at<double>(0);
        lambda[1] = eival.at<double>(1);
        // lambda[0] is the bigger in fabs
        if (fabs(lambda[0]) < fabs(lambda[1])) {
          swap(lambda[0], lambda[1]);
        }

        // TODO: polarity
        if (lambda[0] >= 0) {
          inner_ptr[x] = 0;
        } else {
          double Rb = lambda[1] / lambda[0];
          double S = sqrt(lambda[0]*lambda[0] + lambda[1]*lambda[1]);
          if (max_S_checker < S) max_S_checker = S;
          inner_ptr[x] = exp(-pow(Rb/beta, 2.0)/2) * (1 - exp(-pow((S/gamma), 2.0))/2);
        }
      }
    }
    if (i == 0) {
      result = inner;
    } else if (i == 1) {
      mask8U = inner > result;
      TestUtils::ShowImage(mask8U);
      mask8U.convertTo(mask64F, CV_64FC1);
      result = inner.mul(mask64F);
    } else {
      result = max(result, inner);
    }    
    delta_sigma = sigma * (k_sigma - 1);
    sigma *= k_sigma;
  }
  Mat haha;
  normalize(result, haha, 0, 255, NORM_MINMAX, CV_8U);
  TestUtils::ShowImage(haha);
  TestUtils::Log("max S", max_S_checker);
}