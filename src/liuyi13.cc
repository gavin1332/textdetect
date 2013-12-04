#include "td/liuyi13.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "core/core.h"
#include "utils/test/test.h"

using namespace std;
using namespace cv;

const double LiuYi13::kDoubleEps = 0.0000001;
const uchar LiuYi13::kFG = 255;
const uchar LiuYi13::kBG = 0;
const uchar LiuYi13::kInvalid = 100;
const uchar LiuYi13::kMark = 1;

void LiuYi13::Detect(const cv::Mat& img, std::list<TextRect*>* trlist) {
	extern bool SHOW_FINAL_;

  Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
//  resize(gray, gray, Size(0, 0), 0.5, 0.5);

  Mat resp[2];
	genRespMap(gray, resp);
  
  list<CComp> cclist;
  ParseResp(resp, &cclist);
  
  CheckCCValidation(&cclist);
  
//	cxt.polarity = NEG;
//	HandleOnePolarity(&cxt, trlist);
//
//  if (SHOW_FINAL_) DispRects(cxt.img, *trlist, Scalar(255, 255, 255));
}

void LiuYi13::ParseResp(const Mat& gray, const Mat* resp, list<CComp>* cclist) {
  const double kBinaryThres = 3;
  Mat mask[2];
  mask[POS] = resp[POS] < -kBinaryThres;
  mask[NEG] = resp[NEG] > kBinaryThres;
  TestUtils::ShowImage(mask[POS]);
  TestUtils::ShowImage(mask[NEG]);
  
	const char kFill = 123;
  for (int i = 0; i < 2; ++i) {
    MatIterator_<uchar> itr = mask[i].begin<uchar>();
    MatIterator_<uchar> end = mask[i].end<uchar>();
    for (; itr != end; ++itr) {
      if (*itr != kFG) continue;

      Rect rect;
      floodFill(mask, itr.pos(), kFill, &rect, 0, 0, 8);
      CComp cc;
      for (int y = 0; y < rect.height; ++y) {
        uchar* mask_ptr = mask[i].ptr<uchar>(rect.y + y, rect.x);
        uchar* gray_ptr = gray.ptr<uchar>(rect.y + y, rect.x);
        double* resp_ptr = resp[i].ptr<double>(rect.y + y, rect.x);
        for (int x = 0; x < rect.width; ++x) {
          if (mask_ptr[x] == kFill) {
            cc.AddPix(Pix(rect.x + x, rect.y + y, gray_ptr[x], resp_ptr[x]));
          }
        }
      }
      cclist->push_back(cc);
      floodFill(mask, itr.pos(), kBG, &rect, 0, 0, 8);
    }
  }
}

void LiuYi13::CheckCCValidation(list<CComp>* cclist) {
	CCompItr cc = cclist->begin();
  CCompItr end = cclist->end();
	for (; cc != end; ++cc) {
		if (cc->height < 10 || cc->PixNum() < 100) {
			cc->valid = false;
			continue;
		}
		cc->CalcProperties();
		cc->CheckValidation();
	}
}

void CComp::CheckValidation() {
	if (!valid) return;
	if (height < 10 || height > 200) valid = false;
	if (aspect_ratio > 1.5f || aspect_ratio < 0.1f) valid = false;
	if (area_ratio < 0.2f || area_ratio > 0.65f) valid = false;
	for (int i = 0; i < kCrossVarNum; ++i) {
		if (hcross[i] > 5 || vcross[i] > 5) {
      valid = false;
      break;
    }
	}
}

void LiuYi13::genRespMap(const Mat& gray, Mat* resp) {
  TestUtils::ShowImage(gray);
  Mat fgray;
  gray.convertTo(fgray, CV_64FC1);
  
  const double kInitW = 3;
  const double kSigmaK = sqrt(2);
  double sigma = kInitW / 2 / sqrt(3.0);
  double delta_sigma = sigma;
  const int kN = 12;
  Context cxt;
  cxt.gauss = fgray.clone();
  Mat accum_map[2], mask[2];
  for (int i = 0; i < 2; ++i) {
    mask[i] = Mat::ones(cxt.gauss.size(), CV_8UC1) * 255;
  }

  const double kMaskThres = 0.8;
  for (int i = 0; i < kN; ++i) {
    cout << "i: " << i << endl;
    cxt.StoreGauss();
    GaussianBlur(cxt.gauss, cxt.gauss, Size(0, 0), delta_sigma);
    if (i >= 1) {
      cxt.dog = cxt.gauss - cxt.last_gauss;
      if (i == 1) {
        accum_map[POS] = cxt.dog.clone();
        accum_map[NEG] = cxt.dog.clone();
      } else {
        accum_map[POS] = min(cxt.dog, accum_map[POS]);
        accum_map[NEG] = max(cxt.dog, accum_map[NEG]);
      }
      accum_map[POS].copyTo(resp[POS], mask[POS]);
      accum_map[NEG].copyTo(resp[NEG], mask[NEG]);
      mask[POS] &= 255 - (accum_map[NEG] > kMaskThres);
      mask[NEG] &= 255 - (accum_map[POS] < -kMaskThres);
      double tmax, tmin;
      Point pmax, pmin;
      minMaxLoc(cxt.dog, &tmin, &tmax, &pmin, &pmax);
      cout << "max resp: " << tmin << " " << tmax << endl;
      cout << "pos: " << pmin << " " << pmax << endl;
    }
    delta_sigma = sigma * (kSigmaK - 1);
    sigma *= kSigmaK;
  }
}


void CComp::CalcProperties() {
	int size = pix_vec.size();
	area_ratio = (double) size / (width * height);
	aspect_ratio =	(double) width / height;

	CalcMedianGray();
	CalcGrayStdDev();
	CalcMaskProperties();
}

void CComp::CalcMedianGray() {
  int n = (PixNum() - 1) / 2;
  PixItr begin = pix_vec->begin();
  std::nth_element(begin, begin + n, pix_vec->end(), PixGrayCompare);
  median_gray = (begin + n)->gray_value;
}

void CComp::CalcGrayStdDev() {
	double mean = 0;
  PixItr itr = pix_vec->begin();
  PixItr end = pix_vec->end();
	for (; itr != end; ++itr) {
		mean += itr->gray_value;
	}
	mean /= pix_vec->size();

  PixItr itr = pix_vec->begin();
	for (; itr != end; ++itr) {
    double factor = itr->gray_value - mean;
		gray_stddev += factor * factor;
	}
	gray_stddev = sqrt(gray_stddev / pix_vec->size());
}

void CComp::CalcMaskProperties() {
	Mat mask;
	BuildBorderedMask(&mask);

	perimeter = 0;
	memset(hcross, 0, kCrossVarNum * sizeof(int));
	memset(vcross, 0, kCrossVarNum * sizeof(int));

	int ori_width = width;
	int ori_height = height;
	int step = mask.step1();
	for (int y = 0; y < ori_height; ++y) {
		uchar* ptr = mask.ptr(y + 1) + 1;
		for (int x = 0; x < ori_width; ++x, ++ptr) {
			if (*ptr == kFG) {
				if (*(ptr - step - 1) == kBG || *(ptr - step) == kBG
						|| *(ptr - step + 1) == kBG || *(ptr - 1) == kBG
						|| *(ptr + 1) == kBG || *(ptr + step - 1) == kBG
						|| *(ptr + step) == kBG || *(ptr + step + 1) == kBG) {
					++perimeter;
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

	if (TotalHCross() < 4 && aspect_ratio < 0.3) {
		is_upright_bar_ = true;
	}
}

void CComp::BuildBorderedMask(Mat* mask) {
	*mask = Mat::zeros(Size(width + 2, height + 2), CV_8UC1);
	Point shift(1-x, 1-y);
	PixItr itr = pix_vec.begin();
  PixItr end = pix_vec.end();
	for (; itr != end; ++itr) {
		mask->at<uchar>(itr->pos() + shift) = kFG;
	}
}