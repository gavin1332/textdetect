// Copyright (c) 2010-2011, Tuji
// All rights reserved.
// 
// ${license}
//
// Author: LIU Yi

#ifndef EVALUATE_ICDAR_H_
#define EVALUATE_ICDAR_H_

#include <string>
#include <list>
#include <map>

#include "evaluate/dataset.h"

class ICDAR2005 : public DataSet {
 public:
  ICDAR2005(const std::string& base_dir) : DataSet(base_dir),
  		kTestDataDir(base_dir + "/SceneTrialTest") {
    InitTListMap();
  }
  virtual ~ICDAR2005() {
    ReleaseTListMap();
  }

  void RetrieveTList(const std::string& img_path, std::list<TextRect*>* T_list);

#ifdef TUNE_PARAM
  void TuneParam();
#endif

 private:
  const std::string kTestDataDir;
  std::map<std::string, std::list<TextRect*>*> T_list_map_;

  void InitTListMap();
  void ReleaseTListMap();
};

#endif  // EVALUATE_ICDAR_H_
