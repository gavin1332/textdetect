/*
 * stroke.cc
 *
 *  Created on: Sep 9, 2012
 *      Author: liuyi
 */

#include "td/stroke.h"

#include <opencv2/imgproc/imgproc.hpp>

#include "utils/image/image.h"
#include "utils/test/test.h"

using namespace std;
using namespace cv;

const uchar Stroke::kFG = 255;
const uchar Stroke::kBG = 0;

const float Stroke::kShiftThres = 0.5f;

void Stroke::Extract(const Mat& gray, Mat* level_map) {
  Context cxt;
  Mat temp_koller, temp_level;
  level_map[POS] = Mat::zeros(gray.rows, gray.cols, CV_8UC1);
  level_map[NEG] = Mat::zeros(gray.rows, gray.cols, CV_8UC1);

  const float kKollerThres = 1.7f; //2.f
  for (int level = kMaxLevel; level > 0; --level) {
    float zoom = 1 / pow(2.f, kMaxLevel - level);
    resize(gray, cxt.gray, Size(), zoom, zoom, INTER_LINEAR);
    cxt.level = level;
    BuildScaleSpace(&cxt);
    Koller(&cxt);

    resize(cxt.koller[0], temp_koller, Size(gray.cols, gray.rows));
    level_map[POS] = max(level_map[POS], (temp_koller > kKollerThres) & level);
    level_map[NEG] = max(level_map[NEG], (temp_koller < -kKollerThres) & level);
  }
}

void Stroke::Refine(Mat* mask) {
  Mat element = getStructuringElement(MORPH_RECT, Size(4, 4), Point(1, 1));
  dilate(*mask, *mask, element);

//  const uchar kTemp = 175;
//  const uchar kFill = 100;
//  MatIterator_<uchar> itr = mask->begin<uchar>();
//  MatIterator_<uchar> end = mask->end<uchar>();
//  for (; itr != end; ++itr) {
//    if (*itr == kFG) {
//      Rect rect;
//      floodFill(*mask, itr.pos(), kTemp, &rect, 0, 0, 8);
//      // MODIFIED
//      if (rect.height > 125) {
//        TestToolkit::ShowImage(*mask);
//        ImgToolkit::FillRect(mask, rect, kTemp, kBG);
//        TestToolkit::ShowImage(*mask);
//      } else {
//        ImgToolkit::FillRect(mask, rect, kTemp, kFill);
//      }
//    }
//  }
//
//  *mask = *mask > 0;
}

void Stroke::BuildScaleSpace(Context* cxt) {
  const float kThres = 3.f;
  const float kSigmaK = 2.f;

  const int kLayer = (cxt->level == 1)? 2 : 1;
  const int kGaussLayer = kLayer + 3;
  float sigma[kGaussLayer];
  Mat dog[kGaussLayer-1];
  Mat fgray;
  cxt->gray.convertTo(fgray, CV_32FC1);

//  sigma[0] = cxt->kInitWidth * sqrt((kSigmaK * kSigmaK - 1) / (2 * log(kSigmaK)))
//      / (2 * kSigmaK) / kSigmaK;
  // OPTIMIZE
  sigma[0] = kInitWidth * 0.394f / kSigmaK;
  for (int i = 0; i < kGaussLayer; ++i) {
    if (i > 0) {
      sigma[i] = sigma[i-1] * kSigmaK;
    }
    GaussianBlur(fgray, cxt->gauss[i], Size(0, 0), sigma[i]);

    if (i > 0) {
      dog[i-1] = cxt->gauss[i] - cxt->gauss[i-1];
    }
  }

  for (int i = 0; i < kLayer; ++i) {
    cxt->dog_ext[i] = dog[i+1].clone();
    int width = dog[i].cols;
    int height = dog[i].rows;
    for (int y = 0; y < height; ++y) {
      float* di_ptr = dog[i].ptr<float>(y);
      float* di1_ptr = dog[i+1].ptr<float>(y);
      float* di2_ptr = dog[i+2].ptr<float>(y);
      float* resp_ptr = cxt->dog_ext[i].ptr<float>(y);
      for (int x = 0; x < width; ++x, ++di_ptr, ++di1_ptr, ++di2_ptr, ++resp_ptr) {
        if ((*di1_ptr < kThres || *di1_ptr < *di_ptr || *di1_ptr < *di2_ptr)
            && (*di1_ptr > -kThres || *di1_ptr > *di_ptr || *di1_ptr > *di2_ptr)) {
//        if ((*di1_ptr < kThres || *di1_ptr < *di_ptr)
//            && (*di1_ptr > -kThres || *di1_ptr > *di_ptr)) {
          *resp_ptr = 0;
        }
      }
    }
  }
}

void Stroke::Koller(Context* cxt) {
  const int kLayer = (cxt->level == 1)? 2 : 1;

  int width = cxt->gauss[0].cols;
  int height = cxt->gauss[0].rows;
  Mat resp_mask[kLayer*2];
  for (int i = 0; i < kLayer; ++i) {
    cxt->koller[i] = Mat::zeros(height, width, CV_32FC1);

    int target_width = static_cast<int>(kInitWidth * pow(2.f, i));
    cxt->half_width = target_width / 2;
    resp_mask[2*i + POS] = cxt->dog_ext[i] > FLT_EPSILON;
    Refine(resp_mask + 2*i + POS);
    resp_mask[2*i + NEG] = cxt->dog_ext[i] < -FLT_EPSILON;
    Refine(resp_mask + 2*i + NEG);

    CalcGrad(cxt, i);

    for (int y = 0; y < height; ++y) {
      uchar* pos_ptr = resp_mask[2*i + POS].ptr<uchar>(y);
      uchar* neg_ptr = resp_mask[2*i + NEG].ptr<uchar>(y);
      for (int x = 0; x < width; ++x, ++pos_ptr, ++neg_ptr) {
        if (*pos_ptr == kBG && *neg_ptr == kBG) continue;

        cxt->pos = Point(x, y);
        if (!CalcSideResps(cxt)) continue;

        float pos_koller = min(Positive(cxt->left_resp),
            Positive(cxt->right_resp));
        float neg_koller = max(Negtive(cxt->left_resp),
            Negtive(cxt->right_resp));

        if (pos_koller > 0) {
          cxt->koller[i].at<float>(cxt->pos) = pos_koller;
        } else {
          cxt->koller[i].at<float>(cxt->pos) = neg_koller;
        }
      }
    }

    if (i > 0)  {
      for (int y = 0; y < height; ++y) {
        float* resp_ptr1 = cxt->koller[i-1].ptr<float>(y);
        float* resp_ptr2 = cxt->koller[i].ptr<float>(y);
        for (int x = 0; x < width; ++x, ++resp_ptr1, ++resp_ptr2) {
          if (*resp_ptr1 > -FLT_EPSILON && *resp_ptr2 > *resp_ptr1) {
            *resp_ptr1 = *resp_ptr2;
          } else if (*resp_ptr1 < FLT_EPSILON && *resp_ptr2 < *resp_ptr1) {
            *resp_ptr1 = *resp_ptr2;
          }
        }
      }
    }
  }
}

void Stroke::CalcGrad(Context* cxt, int layer) {
  static const Mat kernel = (Mat_<float>(1, 2) << -1, 1);

  ImgUtils::Gradient(cxt->gauss[layer+1], &cxt->dx, &cxt->dy);
  filter2D(cxt->dx, cxt->dxx, CV_32F, kernel);
  filter2D(cxt->dy, cxt->dyy, CV_32F, kernel.t());
  filter2D(cxt->dx, cxt->dxy, CV_32F, kernel.t());
}

void Stroke::CalcDirection(Context* cxt) {
  cxt->hessian = (Mat_<float>(2, 2) << cxt->dxx.at<float>(cxt->pos),
      cxt->dxy.at<float>(cxt->pos), cxt->dxy.at<float>(cxt->pos),
      cxt->dyy.at<float>(cxt->pos));

  Mat eival, eivec;
  eigen(cxt->hessian, eival, eivec);
  if (fabs(eival.at<float>(0)) > fabs(eival.at<float>(1))) {
    cxt->dir = eivec.row(0).t();
  } else {
    cxt->dir = eivec.row(1).t();
  }
}

bool Stroke::CalcSideResps(Context* cxt) {
  CalcDirection(cxt);

  Point posl, posr;
  posl.x = round(cxt->pos.x - cxt->half_width * cxt->dir.at<float>(0));
  posl.y = round(cxt->pos.y - cxt->half_width * cxt->dir.at<float>(1));
  posr.x = round(cxt->pos.x + cxt->half_width * cxt->dir.at<float>(0));
  posr.y = round(cxt->pos.y + cxt->half_width * cxt->dir.at<float>(1));
  if (!ImgUtils::IsInside(cxt->dx, posl) || !ImgUtils::IsInside(cxt->dx, posr)) {
    return false;
  }

  Mat grad = -(Mat_<float>(2, 1) << cxt->dx.at<float>(posl),
      cxt->dy.at<float>(posl));
  cxt->left_resp = static_cast<float>(grad.dot(cxt->dir));
  grad = (Mat_<float>(2, 1) << cxt->dx.at<float>(posr),
      cxt->dy.at<float>(posr));
  cxt->right_resp = static_cast<float>(grad.dot(cxt->dir));

  return true;
}

void Stroke::CalcExtremalShift(Context* cxt) {
  Mat grad = (Mat_<float>(2, 1) << cxt->dx.at<float>(cxt->pos),
      cxt->dy.at<float>(cxt->pos));
  Mat t = -(grad.t() * cxt->dir) / (cxt->dir.t() * cxt->hessian * cxt->dir);
  cxt->shift.x = t.at<float>(0) * cxt->dir.at<float>(0);
  cxt->shift.y = t.at<float>(0) * cxt->dir.at<float>(1);
}
