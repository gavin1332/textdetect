/*
 * icdar2011.h
 *
 *  Created on: Sep 5, 2012
 *      Author: liuyi
 */

#ifndef EVALUATION_ICDAR2011_H_
#define EVALUATION_ICDAR2011_H_

#include <sstream>

#include "evaluate/dataset.h"

class ICDAR2011 : public DataSet {
 public:
  ICDAR2011(const std::string& base_dir) : DataSet(base_dir),
  		kTestDataDir(base_dir + "/test-textloc-gt/test-textloc-gt") {}
  virtual ~ICDAR2011() {}

  const std::string& test_data_dir() { return kTestDataDir; }

  void RetrieveTList(const std::string& img_path, std::list<TextRect*>* T_list);

 private:
  const std::string kTestDataDir;

  std::string BuildTextRectFilePath(const std::string& img_path);

};

#endif
