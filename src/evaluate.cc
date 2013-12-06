// Copyright (c) 2010-2011, Tuji
// All rights reserved.
// 
// ${license}
//
// Author: LIU Yi

#include "evaluate/evaluate.h"

#include <cmath>

#include "math/math.h"

using namespace std;

const float Evaluator::kAlpha = 0.5f;

void Evaluator::Report(float* precision, float* recall, float* f_measure) {
  *precision = (E_count_ == 0)? 0 : precision_match_accum_ / E_count_;
  *recall = (T_count_ == 0)? 1 : recall_match_accum_ / T_count_;
  if (MathUtils::CmpFloat(*precision, 0) == 0
      || MathUtils::CmpFloat(*recall, 0) == 0) {
    *f_measure = 0;
  } else {
    *f_measure = 1 / (kAlpha / *precision + (1 - kAlpha) / *recall);
  }
}

void Evaluator::RecordMatch(const list<TextRect*>& E_list,
    const list<TextRect*>& T_list) {
  list<TextRect*>::const_iterator it = E_list.begin();
  for (; it != E_list.end(); ++it) {
    precision_match_accum_ += CalcMatch(**it, T_list);
  }
  E_count_ += E_list.size();

  for (it = T_list.begin(); it != T_list.end(); ++it) {
    recall_match_accum_ += CalcMatch(**it, E_list);
  }
  T_count_ += T_list.size();
}

float Evaluator::CalcMatch(const TextRect& input, const TextRect& base) {
  int inner_x1 = max(input.x1(), base.x1());
  int inner_y1 = max(input.y1(), base.y1());
  int inner_x2 = min(input.x2(), base.x2());
  int inner_y2 = min(input.y2(), base.y2());

  if (inner_x1 >= inner_x2 || inner_y1 >= inner_y2) {
    return 0;
  }

  int outer_x1 = min(input.x1(), base.x1());
  int outer_y1 = min(input.y1(), base.y1());
  int outer_x2 = max(input.x2(), base.x2());
  int outer_y2 = max(input.y2(), base.y2());

  float inner_area = (inner_x2 - inner_x1 + 1)*(inner_y2 - inner_y1 + 1);
  float outer_area = (outer_x2 - outer_x1 + 1)*(outer_y2 - outer_y1 + 1);

  return inner_area / outer_area;
}

float Evaluator::CalcMatch(const TextRect& input, const list<TextRect*>& base_list) {
  float match = 0;
  list<TextRect*>::const_iterator it = base_list.begin();
  for (; it != base_list.end(); ++it) {
    float temp = CalcMatch(input, **it);
    match = (temp > match)? temp : match;
  }

  return match;
}
