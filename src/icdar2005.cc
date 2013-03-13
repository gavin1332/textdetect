// Copyright (c) 2010-2011, Tuji
// All rights reserved.
// 
// ${license}
//
// Author: LIU Yi

#include "evaluate/icdar2005.h"

#include <opencv2/highgui/highgui.hpp>

#include "evaluate/evaluate.h"
#include "utils/common/common.h"
#include "utils/xml/tinyxml.h"
#include "utils/test/test.h"

using namespace std;
using namespace cv;

void ICDAR2005::RetrieveTList(const string& img_path, list<TextRect*>* T_list) {
  size_t pos = img_path.find_last_of('/', img_path.find_last_of('/'));
  string id = img_path.substr(pos + 1);

  T_list =  T_list_map_.at(id);
}

void ICDAR2005::InitTListMap() {
  string xml_path = base_dir_ + '/' + kTestDataDir + "/locations.xml";
  TiXmlDocument doc(xml_path.c_str());
  doc.LoadFile();

  TiXmlElement* tagset_elem = doc.FirstChildElement();
  TiXmlElement* image_elem = tagset_elem->FirstChildElement();
  for (; image_elem != NULL; image_elem = image_elem->NextSiblingElement()) {
    string image_name = image_elem->FirstChildElement("imageName")->GetText();

    TiXmlElement* rects_elem = image_elem->FirstChildElement("taggedRectangles");

    list<TextRect*>* rect_list = new list<TextRect*>;
    TiXmlElement* rect_elem = rects_elem->FirstChildElement();
    for (; rect_elem != NULL; rect_elem = rect_elem->NextSiblingElement()) {
      int x = static_cast<int>(atof(rect_elem->Attribute("x")));
      int y = static_cast<int>(atof(rect_elem->Attribute("y")));
      int width = static_cast<int>(atof(rect_elem->Attribute("width")));
      int height = static_cast<int>(atof(rect_elem->Attribute("height")));

      rect_list->push_back(new TextRect(x, y, x + width - 1, y + height - 1));
    }

    T_list_map_.insert(pair<string, list<TextRect*>*>(image_name, rect_list));
  }
}

void ICDAR2005::ReleaseTListMap() {
  map<string, list<TextRect*>*>::iterator mit = T_list_map_.begin();
  for (; mit != T_list_map_.end(); ++mit) {
  	list<TextRect*>::iterator lit = mit->second->begin(), lend = mit->second->end();
  	for (; lit != lend; ++lit) {
  		delete *lit;
  	}
    delete mit->second;
  }
}

#ifdef TUNE_PARAM
void ICDAR2005::TuneParam() {
  for (float kBinThresTario = 0.07f; kBinThresTario  <= 0.13f;
      kBinThresTario += 0.02f) {
    for (float kJumpRatioThres_ = 0.25f; kJumpRatioThres_  <= 0.6f;
        kJumpRatioThres_ += 0.04f) {
      TestToolkit::Print("kBinThresTario", kBinThresTario);
      TestToolkit::Print("kJumpRatioThres_", kJumpRatioThres_);
      StrokeModel::set_kBinThresTario(kBinThresTario);
      StrokeModel::set_kJumpRatioThres(kJumpRatioThres_);
      Evaluate();
      TestToolkit::Print("\n");
    }
  }
}
#endif
