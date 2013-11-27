#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "core/core.h"
#include "evaluate/evaluate.h"
#include "evaluate/icdar2011.h"
#include "td/stroke-filter.h"
#include "utils/common/common.h"
#include "utils/image/image.h"
#include "utils/image/mser.h"
#include "utils/test/test.h"
#include "td/frangi98.h"

using namespace std;
using namespace cv;

extern bool PRINT_LOG_;
extern bool SPLIT_TEXT_RECT_;
extern bool SHOW_EXTREMAL_RESPONSE_;
extern bool SHOW_GROUPED_RESULT_;
extern bool SHOW_GROUP_STEP_;
extern bool SHOW_FINAL_;

bool PRINT_LOG_ = true;
bool SPLIT_TEXT_RECT_ = !true;
bool SHOW_GROUP_STEP_ = !true;
bool SHOW_EXTREMAL_RESPONSE_ = true;
bool SHOW_GROUPED_RESULT_ = true;
bool SHOW_FINAL_ = true;

void SaveEList(const list<TextRect*>& E_list, const String& img_path) {
//  ofstream writer(img_path + ".txt");
//  list<TextRect>::const_iterator itr = E_list.begin();
//  for (; itr != E_list.end(); ++itr) {
//    writer << itr->x1() << " " << itr->y1() << " " << itr->x2() << " " << itr->y2() << endl;
//  }
//  writer.close();
//  TestToolkit::Print("Saved estimated rectangles\n");
}

void ReadRects(const String& img_path, vector<Rect>* rect_vec) {
  ifstream in(img_path + ".txt");
  if (in.fail()) return;

  vector<string> piece_vec;
  string line;
  while (true) {
    getline(in, line);
    if (line.length() == 0) break;

    piece_vec.clear();
    CmnUtils::Split(line, ' ', &piece_vec);

    int x1 = atoi(piece_vec[0].c_str());
    int y1 = atoi(piece_vec[1].c_str());
    int x2 = atoi(piece_vec[2].c_str());
    int y2 = atoi(piece_vec[3].c_str());
    Rect rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1);

    rect_vec->push_back(rect);
  }

  in.close();
}

void ReleaseList(list<TextRect*>* trlist) {
	list<TextRect*>::iterator it = trlist->begin(), end = trlist->end();
	for (; it != end; ++it) {
		delete *it;
	}
	trlist->clear();
}

void PrintProgress(int count, int size) {
  cout << "Progress:" << (float) count*100/size << "%" << endl;
}

#if 0

int main(int argc, char** argv) {
  const string base_dir = "/home/liuyi/project/cpp/testdata/scene/2011";
  ICDAR2011 icdar2011(base_dir);

  const string data_dir = icdar2011.test_data_dir();
//  const string data_dir = base_dir + "/test-textloc-gt/hard";
  vector<string> filename_vec;
  CmnUtils::RetrieveFilenames(data_dir, ".jpg", &filename_vec, true);
  sort(filename_vec.begin(), filename_vec.end());

  Evaluator evaluator;
  vector<string>::iterator it = filename_vec.begin();
  int count = 1;
  const int file_count_total = filename_vec.size();

  StrokeFilter detector;
  double exec_time = (double) getTickCount();
  for (; it != filename_vec.end(); ++it, ++count) {
    if (it->compare("124.jpg") < 0) continue;

    const string img_path = data_dir + "/" + *it;
    TestUtils::Print(img_path);
    PrintProgress(count, file_count_total);

    Mat img = imread(img_path, CV_LOAD_IMAGE_COLOR);

    vector<Rect> rect_vec;
    ReadRects(img_path, &rect_vec);
    bool notified = TestUtils::ShowRects(img, rect_vec, Scalar(255, 255, 255));
    if (notified) continue;

    float zoom = 1;
    if (img.cols > 1280) {
    	zoom = (float) 1280 / img.cols;
    	resize(img, img, Size(), zoom, zoom);
    }

//    Rect rect;
//    bool success = TestUtils::UserDrawRect(img, &rect);
//    if (!success) continue;
//    img = img(rect);

    list<TextRect*> T_list;
    list<TextRect*> E_list;
    icdar2011.RetrieveTList(img_path, &T_list);
    detector.Detect(img, &E_list);
    if (zoom < 1) {
    	list<TextRect*>::iterator it = E_list.begin(), end = E_list.end();
    	for (; it != end; ++it) {
    		(*it)->TuneRect(1 / zoom);
    	}
    }

    evaluator.RecordMatch(E_list, T_list);

    SaveEList(E_list, img_path);
    ReleaseList(&T_list);
    ReleaseList(&E_list);
  }

  exec_time = ((double) getTickCount() - exec_time) / getTickFrequency();
  TestUtils::Print("Time cost/s", exec_time);

  float precision;
  float recall;
  float f_measure;
  evaluator.Report(&precision, &recall, &f_measure);

  TestUtils::Print("precision", precision);
  TestUtils::Print("recall", recall);
  TestUtils::Print("f_measure", f_measure);

  return EXIT_SUCCESS;
}

#elif 1

int main() {
  const string base_dir = "/home/liuyi/project/cpp/testdata/scene/2011";
  const string img_path = base_dir + "/test-textloc-gt/test-textloc-gt/153.jpg";
  Mat img = imread(img_path, CV_LOAD_IMAGE_COLOR);
  Frangi98 detector;
  list<TextRect*> result;
  detector.Detect(img, &result);
  return 0;
}

#else

int main() {
  Mat a(3, 4, CV_64FC1, 0.0);
  Mat b(3, 4, CV_64FC1, 1.0);
  Mat c = b > a;
  cout << c << endl;
  return 0;
}

#endif
