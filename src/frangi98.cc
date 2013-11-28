#include "td/frangi98.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "core/core.h"
#include "utils/test/test.h"

using namespace std;
using namespace cv;

void Frangi98::Detect(const cv::Mat& img, std::list<TextRect*>* trlist) {
	extern bool SHOW_FINAL_;

  Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
//  resize(gray, gray, Size(0, 0), 0.5, 0.5);

	HandleOnePolarity(gray, trlist);
//	cxt.polarity = NEG;
//	HandleOnePolarity(&cxt, trlist);
//
//  if (SHOW_FINAL_) DispRects(cxt.img, *trlist, Scalar(255, 255, 255));
}

void Frangi98::HandleOnePolarity(const Mat& gray, std::list<TextRect*>* trlist) {
  const int width = gray.cols;
  const int height = gray.rows;
  TestUtils::ShowImage(gray);
  Mat fgray;
  gray.convertTo(fgray, CV_64FC1);
  
  const double init_w = 10;
  const double k_sigma = sqrt(2);
  double sigma = init_w / 2 / sqrt(3.0);
  double delta_sigma = sigma;
  Mat gauss = fgray;
  const int N = 6;
//  const double beta = 0.5;
//  // TODO: need to tune, sensitive
//  const double gamma = 25;
  // TODO: ksize can significantly affect the result
  const int ksize = 3;
  // refers to the document of getDerivKernels: the coefficients should have 
  // the denominator= pow(2.0, ksize*2-dx-dy-2)
  const double normal = 1 / pow(2.0, (double) ksize*2 - 4);
  double lambda[2];
  Mat dxx, dyy, dxy;
  Mat hessian(2, 2, CV_64FC1);
  Mat inner(gray.size(), CV_64FC1, 0.0);
  Mat result[2], mask[2];
  for (int i = 0; i < 2; ++i) {
    result[i] = Mat::zeros(gray.size(), CV_64FC1);
    mask[i] = Mat::zeros(gray.size(), CV_64FC1);
  }
  double max_S_checker = 0;
  for (int i = 0; i < N; ++i) {
    cout << "i: " << i << endl;
    if (delta_sigma < 1.0) {
      GaussianBlur(fgray, gauss, Size(0, 0), sigma);
    } else {
      GaussianBlur(gauss, gauss, Size(0, 0), delta_sigma);
    }
    // should we consider the smooth effect of Sobel, which may affect hessian
    // normalization
    Sobel(gauss, dxx, CV_64F, 2, 0, ksize, normal, 0, BORDER_REPLICATE);
    Sobel(gauss, dyy, CV_64F, 0, 2, ksize, normal, 0, BORDER_REPLICATE);
    Sobel(gauss, dxy, CV_64F, 1, 1, ksize, normal, 0, BORDER_REPLICATE);
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
        // normalization
        hessian *= sigma * sigma;

        Mat eival, eivec;
        eigen(hessian, eival, eivec);
        lambda[0] = eival.at<double>(0);
        lambda[1] = eival.at<double>(1);
        // lambda[0] is the bigger in fabs
        if (fabs(lambda[0]) < fabs(lambda[1])) {
          swap(lambda[0], lambda[1]);
        }

        double Rb = fabs(lambda[1] / lambda[0]);
        double S = sqrt(lambda[0]*lambda[0] + lambda[1]*lambda[1]);
        if (max_S_checker < S) max_S_checker = S;
//        if (lambda[0] >= 0) {
//          inner_ptr[x] = 0;//-exp(-pow(Rb/beta, 2.0)/2) * (1 - exp(-pow((S/gamma), 2.0))/2);
//        } else {
//          inner_ptr[x] = exp(-pow(Rb/beta, 2.0)/2) * (1 - exp(-pow((S/gamma), 2.0))/2);
//        }
        const double Rb_thres = 0.9;
        const double S_thres = 23.0;
        inner_ptr[x] = (Rb > Rb_thres || S < S_thres)? 0 : lambda[0];
      }
    }
    result[POS] = min(result[POS], inner);
    result[POS].copyTo(mask[NEG]);
    result[NEG] = max(result[NEG], inner);
    result[NEG].copyTo(mask[POS]);
    delta_sigma = sigma * (k_sigma - 1);
    sigma *= k_sigma;
  }
//  normalize(result, output, 0, 255, NORM_MINMAX, CV_8U);
  const double epsilon = 0.0000001;
  TestUtils::ShowImage(result[POS] < -epsilon);
  TestUtils::ShowImage(result[NEG] > epsilon);
//  TestUtils::ShowImage(output > 135); // 不知道为什么阈值设为135和125差别碰巧会很大
  TestUtils::Log("max S", max_S_checker);
}