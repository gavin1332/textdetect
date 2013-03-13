// Copyright (c) 2010-2011, Tuji
// All rights reserved.
// 
// ${license}
//
// Author: LIU Yi

#ifndef EVALUATE_EVALUATE_H_
#define EVALUATE_EVALUATE_H_

#include <list>

#include "core/core.h"

class Evaluator {
 public:
  Evaluator() : precision_match_accum_(0), recall_match_accum_(0), E_count_(0),
      T_count_(0) {}
  virtual ~Evaluator() {}

  void Clear() {
    precision_match_accum_ = 0;
    recall_match_accum_ = 0;
    E_count_ = 0;
    T_count_ = 0;
  }

  void Report(float* precision, float* recall, float* f_measure);

  void RecordMatch(const std::list<TextRect*>& E_list,
      const std::list<TextRect*>& T_list);

 private:
  static const float kAlpha;

  float precision_match_accum_;
  float recall_match_accum_;
  int E_count_;
  int T_count_;

  float CalcMatch(const TextRect& input, const TextRect& base);

  float CalcMatch(const TextRect& input, const std::list<TextRect*>& base_list);

};

#endif  // EVALUATE_EVALUATE_H_
