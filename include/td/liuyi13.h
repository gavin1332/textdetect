#ifndef TD_LIUYI13_H_
#define	TD_LIUYI13_H_

#include <list>

#include <opencv2/core/core.hpp>

#include "text-detect.h"

class LiuYi13 : public TextDetector {
public:
  LiuYi13() {}
  virtual ~LiuYi13() {}
  
	void Detect(const cv::Mat& img, std::list<TextRect*>* trlist);
  
private:
  typedef std::vector<CComp>::iterator CCompItr;
  
  static const double kDoubleEps;
  static const uchar kFG;
  static const uchar kBG;
  static const uchar kInvalid;
  static const uchar kMark;
  
  typedef struct context {
    cv::Mat last_gauss;
    cv::Mat gauss;
    cv::Mat dog;
    
    void StoreGauss() {
      last_gauss = gauss.clone();
    }
  } Context;

//  void Handle(const cv::Mat& gray, std::list<TextRect*>* trlist);
  
  void ParseResp(const cv::Mat& gray, const cv::Mat* resp,
      std::list<CComp>* cclist);
  
  void CheckCCValidation(std::list<CComp>* cclist);
  
  void genRespMap(const cv::Mat& gray, cv::Mat* resp);
  
  void GroupRegion(const cv::Mat& binary, std::vector<Region*>* region_vec,
  		std::list<TextRect*>* trlist);
  
  LiuYi13(const LiuYi13&);
  void operator=(const LiuYi13&);

};

class Pix {
public:
  const int x;
  const int y;
  uchar gray_value;
  double resp_value;
  
  Pix(int x, int y, uchar gray_value, double resp_value) : x(x), y(y),
      gray_value(gray_value) , resp_value(resp_value) {}
  
  cv::Point pos() const {
    return cv::Point(x, y);
  }
};

// connected component
class CComp {
public:
  const int x, y, width, height;
  std::vector<Pix>* pix_vec;
  bool valid;
  
  CComp(int x, int y, int width, int height) : x(x), y(y), width(width),
      height(height), valid(true) {
    pix_vec = new std::vector<Pix>;
  }
  virtual ~CComp() {
    delete pix_vec;
  }
  
  cv::Rect ToCvRect() const { return cv::Rect(x, y, width, height); }
  
  AddPix(Pix pix) {
    pix_vec->push_back(pix);
  }
  
  int PixNum() const {
    return static_cast<int>(pix_vec->size());
  }
  
	void CalcProperties();
	void CheckValidation();
  
private:
  typedef std::vector<Pix>::iterator PixItr;

	static const int kCrossVarNum = 3;
  
  double area_ratio;
  double aspect_ratio;
  uchar median_gray;
  double gray_stddev;
  
	bool is_upright_bar_;
	int perimeter;
	int hcross[kCrossVarNum];
	int vcross[kCrossVarNum];
  
  static bool PixGrayCompare(const Pix& p1, const Pix& p2) {
    return p1.gray_value < p2.gray_value;
  }
  void CalcMedianGray();
  void CalcGrayStdDev();
	void CalcMaskProperties();
  void BuildBorderedMask(cv::Mat* mask);
	int TotalHCross() const { return hcross[0] + hcross[1] + hcross[2]; }
	int TotalVCross() const { return vcross[0] + vcross[1] + vcross[2]; }
}; 

#endif

