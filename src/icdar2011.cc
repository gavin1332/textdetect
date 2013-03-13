/*
 * icdar2011.cc
 *
 *  Created on: Sep 5, 2012
 *      Author: liuyi
 */

#include "evaluate/icdar2011.h"

#include <cstring>
#include <cstdlib>

#include <vector>
#include <fstream>

#include "utils/common/common.h"

using namespace std;

void ICDAR2011::RetrieveTList(const string& img_path, list<TextRect*>* T_list) {
  string path = BuildTextRectFilePath(img_path);

  vector<string> piece_vec;
  ifstream in(path.c_str());
  string line;
  while (true) {
    getline(in, line);
    if (line.length() == 0) break;

    piece_vec.clear();
    CmnUtils::Split(line, ',', &piece_vec);

    int x1 = atoi(piece_vec[0].c_str());
    int y1 = atoi(piece_vec[1].c_str());
    int x2 = atoi(piece_vec[2].c_str());
    int y2 = atoi(piece_vec[3].c_str());
    string text = piece_vec[4].substr(1, piece_vec[4].length() - 2);
    TextRect* tr = new TextRect(x1, y1, x2, y2, text);

    T_list->push_back(tr);
  }
}

string ICDAR2011::BuildTextRectFilePath(const string& img_path) {
  string filename = img_path;
  size_t pos = img_path.find_last_of('/');
  if (pos != string::npos) {
    filename = img_path.substr(pos + 1);
  }
  vector<string> piece_vec;
  CmnUtils::Split(filename, '.', &piece_vec);

  stringstream ss;
  ss << kTestDataDir << "/gt_" << piece_vec[0] << ".txt";

  return ss.str();
}
