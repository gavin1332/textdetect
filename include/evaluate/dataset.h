/*
 * dataset.h
 *
 *  Created on: Sep 5, 2012
 *      Author: liuyi
 */

#ifndef EVALUATE_DATASET_H_
#define EVALUATE_DATASET_H_

#include <string>
#include <list>

#include "core/core.h"

class DataSet {
 public:
  DataSet(const std::string& base_dir) : base_dir_(base_dir) {}
  virtual ~DataSet() {}

  virtual void RetrieveTList(const std::string& img_path,
      std::list<TextRect*>* T_list) = 0;

 protected:
  std::string base_dir_;

};

#endif /* EVALUATE_DATASET_H_ */
