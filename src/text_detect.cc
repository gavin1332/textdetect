/*
 * text_detect.cc
 *
 *  Created on: Sep 5, 2012
 *      Author: liuyi
 */

#include "td/text_detect.h"

#include <sys/stat.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "td/stroke.h"
#include "utils/image/image.h"
#include "utils/test/test.h"

using namespace std;
using namespace cv;

const uchar TextDetector::kFG = 255;
const uchar TextDetector::kBG = 0;
const uchar TextDetector::kInvalid = 100;
const uchar TextDetector::kMark = 1;

const float TextDetector::kFloatEps = 0.000001;

void TextDetector::Detect(const Mat& img, list<TextRect*>* trlist) {
	extern bool SHOW_FINAL_;

  Context cxt;
	cxt.img = img;

	cvtColor(img, cxt.gray, CV_BGR2GRAY);

	Stroke stroke;
	stroke.Extract(cxt.gray, cxt.level_map);

	cxt.polarity = POS;
	HandleOnePolarity(&cxt, trlist);
	cxt.polarity = NEG;
	HandleOnePolarity(&cxt, trlist);

  OverlapAnalyse(trlist);
//  Split(trlist);

  if (SHOW_FINAL_) DispRects(cxt.img, *trlist, Scalar(255, 255, 255));
}

void TextDetector::HandleOnePolarity(Context* cxt, list<TextRect*>* trlist) {
	extern bool SHOW_GROUPED_RESULT_;

//	RemoveNoise(cxt);
	bool notified = TestUtils::ShowImage(*cxt->CurrLevelMap() * 60);
	if (notified) return;

	RegionTree<ConnComp> region_tree(*cxt->CurrLevelMap(), 4, true);
	region_tree.Parse();

//	TestUtils::ShowImage((*cxt->CurrLevelMap() > 3) * 60);
//	TestUtils::ShowImage((*cxt->CurrLevelMap() > 2) * 60);
//	TestUtils::ShowImage((*cxt->CurrLevelMap() > 1) * 60);
//	TestUtils::ShowImage((*cxt->CurrLevelMap() > 0) * 60);

	vector<Region*>* region_pool = region_tree.region_pool();
	CheckCCValidation(cxt, region_pool);
	ShowValidCCs(cxt, region_pool->begin(), region_pool->end());

	CheckCCRelation(cxt, region_pool);
	ShowValidCCs(cxt, region_pool->begin(), region_pool->end());

	vector<Region*> region_vec;
	for (RegionItr it = region_pool->begin(); it != region_pool->end(); ++it) {
		ConnComp* cc = static_cast<ConnComp*>(*it);
		if (cc->is_valid()) {
			region_vec.push_back(*it);
		}
	}

	list<TextRect*> tmplist;
	GroupRegion(cxt, &region_vec, &tmplist);
	Refine(cxt, &tmplist);

  if (SHOW_GROUPED_RESULT_) DispRects(*cxt->CurrLevelMap()*60, tmplist, Scalar(255, 255, 255));

	trlist->splice(trlist->end(), tmplist);
}

void TextDetector::CheckCCValidation(Context* cxt, vector<Region*>* region_pool) {
	RegionItr it = region_pool->begin(), end = region_pool->end();
	for (; it != end; ++it) {
		ConnComp* cc = static_cast<ConnComp*>(*it);
		if (cc->Height() < 10) {
			cc->Invalidate();
			continue;
		}
		cc->CalcProperties(cxt->gray);
		cc->CheckValidation();
	}
}

void TextDetector::CheckParentCC(Context* cxt, vector<Region*>* region_pool) {
	RegionItr it = region_pool->begin(), last = region_pool->end() - 1;
	for (; it != last; ++it) {
		ConnComp* cc = static_cast<ConnComp*>(*it);
		if (!cc->is_valid()) continue;

		ConnComp* parent = static_cast<ConnComp*>(cc->parent());
		if (parent->IsRoot() || !parent->is_valid()) continue;

		if (IsSimilarHeight(cc, parent)) {
			if (IsSimilarWidth(cc, parent)) {
				parent->Invalidate();
//				if (cc->area_ratio() > parent->area_ratio()) {
//					cc->Invalidate();
//				} else {
//					parent->Invalidate();
//				}
				cout << 1 << endl;
			} else {
				if (IsSimilarStdDev(cc, parent) && (parent->aspect_ratio() < 1 ||
						cc->is_upright_bar())) {
					cc->Invalidate();
				} else {
					parent->Invalidate();
				}
				cout << 2 << endl;
			}
		} else {
			if (IsSimilarWidth(cc, parent)) {
				if (cc->is_upright_bar() || IsSimilarStdDev(cc, parent)
						|| (cc->level() == Stroke::kMaxLevel && cc->Height() > 20)) {
					cc->Invalidate();
				} else {
					parent->Invalidate();
				}
				cout << 3 << endl;
			} else {
				if (cc->is_upright_bar() || IsSimilarStdDev(cc, parent)) {
					cc->Invalidate();
				} else {
					parent->Invalidate();
				}
				cout << 4 << endl;
			}
		}
//		ShowValidCCs(cxt, region_pool->begin(), region_pool->end());
	}
}

void TextDetector::CheckCCRelation(Context* cxt, vector<Region*>* region_pool) {
	CheckParentCC(cxt, region_pool);
//	ShowValidCCs(cxt, region_pool->begin(), region_pool->end());

	CheckAncestorCC(cxt, region_pool);
}

void TextDetector::CheckAncestorCC(Context* cxt, vector<Region*>* region_pool) {
	RegionItr it = region_pool->begin(), last = region_pool->end() - 1;
	for (it = region_pool->begin(); it != last; ++it) {
		ConnComp* cc = static_cast<ConnComp*>(*it);
		if (!cc->is_valid()) continue;

		ConnComp* parent = static_cast<ConnComp*>(cc->parent());
		while (!parent->IsRoot()) {
			if (parent->is_valid()) {
				if (cc->IsMoreInformativeThan(parent) ||
						(!cc->is_upright_bar() && parent->aspect_ratio() > 2)) {
					parent->Invalidate();
				} else {
					cc->Invalidate();
					break;
				}
			}
			parent = static_cast<ConnComp*>(parent->parent());
		}
	}
}

void TextDetector::RemoveNoise(Context* cxt) {
	Mat* level_map = cxt->CurrLevelMap();
	Mat mask = *level_map > 0;

	const char kFill = 123;
	MatIterator_<uchar> itr = mask.begin<uchar>();
	MatIterator_<uchar> end = mask.end<uchar>();
	for (; itr != end; ++itr) {
		if (*itr != kFG)
			continue;

		Rect rect;
		int pixnum = floodFill(mask, itr.pos(), kFill, &rect, 0, 0, 8);

		if (pixnum < 100) { //rect.width < 8 || rect.height < 8) {
			floodFill(mask, itr.pos(), kBG, &rect, 0, 0, 8);
		}
	}

	*level_map = (mask > 0) & *level_map;
}

void TextDetector::ShowValidCCs(Context* cxt, RegionItr begin, RegionItr end) {
	Mat map;
	BuildMapWithValidCCs(cxt, &map, begin, end);
	TestUtils::ShowImage(map * 60);
}

void TextDetector::BuildMapWithValidCCs(Context* cxt, Mat* map,
		RegionItr begin, RegionItr end) {
	*map = Mat::zeros(cxt->gray.size(), CV_8UC1);
	vector<Region*>::reverse_iterator rit(end), rend(begin);
	for (; rit != rend; ++rit) {
		ConnComp* cc = static_cast<ConnComp*>(*rit);
		if (cc->is_valid()) {
			vector<Pixel*>::const_iterator pit = cc->pix_vec().begin();
			vector<Pixel*>::const_iterator pend = cc->pix_vec().end();
			for (; pit != pend; ++pit) {
				map->at<uchar>((*pit)->pos()) = cc->level();
			}
		}
	}
}

void TextDetector::GroupRegion(Context* cxt, vector<Region*>* region_vec,
		list<TextRect*>* trlist) {
	extern bool SHOW_GROUP_STEP_;

	RegionItr begin = region_vec->begin(), end = region_vec->end();
	sort(begin, end, CompareY1);

	Mat map;
	BuildMapWithValidCCs(cxt, &map, begin, end);
	map *= 60;

	const uchar kLooseThres = 35;
	bool show_group_step = SHOW_GROUP_STEP_;

	RegionItr regbase = begin;
	while (regbase != end) {
		ConnComp* ccbase = static_cast<ConnComp*>(*regbase);
		TextRect* tr = new TextRect(ccbase);
		ccbase->Invalidate();

		RegionItr it = regbase;
		for (++it; it != end; ++it) {
			ConnComp* cc= static_cast<ConnComp*>(*it);
			if (!IsInSearchScope(tr, cc)) break;

//			const int kDistThres = cc->Width() * 3 / 2;
			const int kDistThres = cc->Width() / 2;
			if (!cc->is_valid() || tr->DistHorizon(cc) > kDistThres) continue;

			if (show_group_step) {
				show_group_step &= !TestUtils::ShowRect(map, tr->ToCvRect(), Scalar(0, 0, 255));
			}
			if (show_group_step) {
				show_group_step &= !TestUtils::ShowRect(map, cc->ToCvRect(), Scalar(0, 255, 0));
			}

			bool goon = true;
			uchar median_gray_diff = abs(cc->median_gray() - tr->median_gray_mean());
			TestUtils::Log("GroupRect - median_gray_diff", (int) median_gray_diff);
			if (CheckHorizon(tr, cc)) {
				goon = false;
				TestUtils::Log("GroupRect - CheckHorizon", true);
			} else if (tr->Contains(cc) && median_gray_diff < kLooseThres) {
				goon = false;
				TestUtils::Log("GroupRect - base->Contains(cc) && median_gray_diff < kLooseThres");
			}

			if (goon) {
				continue;
			}

			tr->AddConnComp(cc);
			cc->Invalidate();
			it = regbase;

			TestUtils::Log("GroupRect - AddRegion\n");
			if (show_group_step) {
				show_group_step &= !TestUtils::ShowRect(map, tr->ToCvRect(), Scalar(255, 0, 0));
			}
		}

		if (tr->CharNum() > 1) {
			trlist->push_back(tr);
		} else {
			delete tr;
		}

		for(++regbase; regbase != end && !static_cast<ConnComp*>(*regbase)->is_valid();
				++regbase) {}
	}
}

bool TextDetector::CheckHorizon(const TextRect* tr, const ConnComp* region) {
  int my1, my2;
  tr->MedianY12(&my1, &my2);
  int mh = my2 - my1;

  bool top_or_bottom_match = min(abs(region->y1() - my1), abs(region->y2() - my2))
      < (my2 - my1) / 3 || abs(region->y1() - tr->y1()) < (my2 - my1) / 3
      || abs(region->y2() - tr->y2()) < (my2 - my1) / 3;
//  bool top_or_bottom_match = min(abs(cc->y1() - my1), abs(cc->y2() - my2))
//      < (my2 - my1) / 3;
  float h_max_h_ratio = (float) std::min(region->Height(), mh)
      / std::max(region->Height(), mh);
  if (h_max_h_ratio > 0.5f && top_or_bottom_match) return true;
  if (tr->NeighborRegion(region)->MatchY(region)) return true;
  if (tr->CharNum() > 3) {
    int median_h = tr->MedianHeight();
    float h_median_h_ratio = (float) std::min(region->Height(), median_h)
        / std::max(region->Height(), median_h);
    return h_median_h_ratio > 0.75f && top_or_bottom_match;
  }
  return false;
}

void TextDetector::DispRects(const Mat& gray, const list<TextRect*>& trlist,
    Scalar color) {
  vector<Rect> rect_vec;
  list<TextRect*>::const_iterator itr = trlist.begin(), end = trlist.end();
  for (; itr != end; ++itr) {
    rect_vec.push_back((*itr)->ToCvRect());
  }
  TestUtils::ShowRects(gray, rect_vec, color);
}

void TextDetector::OverlapAnalyse(list<TextRect*>* trlist) {
  int inner_x1, inner_y1, inner_x2, inner_y2;
  list<TextRect*>::iterator it = trlist->begin();
  while (it != trlist->end()) {
  	TextRect* tri = *it;
    bool valid = true;
    list<TextRect*>::iterator jt = trlist->begin();
    for (; jt != trlist->end(); ++jt) {
      if (jt == it) continue;
      TextRect* trj = *jt;

      inner_x1 = max(tri->x1(), trj->x1());
      inner_y1 = max(tri->y1(), trj->y1());
      inner_x2 = min(tri->x2(), trj->x2());
      inner_y2 = min(tri->y2(), trj->y2());
      if (inner_x1 >= inner_x2 || inner_y1 >= inner_y2) continue;

      int inner_area = (inner_x2 - inner_x1 + 1)*(inner_y2 - inner_y1 + 1);
      float iarea_ratio = (float) inner_area / tri->Area();
//      float jarea_ratio = (float) inner_area / trj->Area();

//      TestUtils::ShowRect(state_.input_gray, itr->CvRect());
//      TestUtils::ShowRect(state_.input_gray, jtr->CvRect());
//      // Keep the smaller as polarity is the same, or keep the bigger
//      if (iarea_ratio > 0.85f) {
//        if (tri->Area() < trj->Area() && jarea_ratio < 0.85f) {
//          valid = false;
//          it = trlist->erase(it);
//          break;
//        }
//      }
      // TODO: temp
      if (iarea_ratio > 0.85f) {
        if (tri->Width() < trj->Width()) {
          valid = false;
          it = trlist->erase(it);
          break;
        }
      }
    }

    if (valid) ++it;
  }
}

void TextDetector::Refine(Context* cxt, list<TextRect*>* trlist) {
  list<TextRect*>::iterator it = trlist->begin();
  while (it != trlist->end()) {
  	TextRect* tr = *it;
    bool erasable = false;
    if (tr->CharNum() < 3) {
      if (tr->CharNum() == 2) {
        TextRect front = tr->FrontChar();
        TextRect back = tr->BackChar();
        if (abs(front.y1() - back.y1()) + abs(front.y2() - back.y2()) > tr->Height() / 5
            || front.Width() + back.Width() < 15) {
          erasable = true;
        }
      } else {
        erasable = true;
      }
      TestUtils::Log("Refine - itr->CharNum()", tr->CharNum());
//    } else if ((cxt->level() == 0 && itr->MedianWidth() < 6)
//        || (state_.level > 1 && itr->MedianWidth() < 8)) {
//      erasable = true;
//      TestUtils::Log("Refine - itr->MedianWidth()", itr->MedianWidth());
    } else if ((float) tr->Width() / tr->Height() < 1.45f) {
      erasable = true;
      TestUtils::Log("Refine - itr->Width() / itr->Height()", tr->Width() / tr->Height());
    }

//    int gray_num = 0;
//    for (int y = itr->y1(); y < itr->y2(); ++y) {
//      const uchar* ptr = stroke.ptr(y, itr->x1());
//      for (int x = itr->x1(); x < itr->x2(); ++x, ++ptr) {
//        if (*ptr == kGray) ++gray_num;
//      }
//    }
//    if (gray_num > itr->Width() / 3) {
//      erasable = true;
//      Print("Refine - gray_num", gray_num);
//      Print("Refine - itr->Width() / 3", itr->Width() / 3);
//    }

    if (erasable) {
      it = trlist->erase(it);
    } else {
      ++it;
    }
  }
  TestUtils::Log("\n");
}


void TextDetector::Split(list<TextRect*>* trlist) {
  list<TextRect*>::iterator it = trlist->begin();
  while (it != trlist->end()) {
    int sub_rect_num = (*it)->CharNum();
    if (sub_rect_num < 4) {
      ++it;
      continue;
    }

    int median_interval = (*it)->MedianInterval();
    int median_width = (*it)->MedianWidth();
    float wthres;
    // OPTIMIZE
    if (median_interval < 2) {
      wthres = (float) median_width / 2;
    } else {
      wthres = median_interval + median_width * 0.5f;
    }

    const list<ConnComp*>& cclist = (*it)->cc_list();
    list<ConnComp*>::const_iterator end = cclist.end();
    list<ConnComp*>::const_iterator head = cclist.begin();
    list<ConnComp*>::const_iterator ait = head;
    for (; (*ait)->width_contained(); ++ait) {}

    list<ConnComp*>::const_iterator bit = ait;
    for (; ait != end; ait = bit) {
      for (++bit; bit != end && (*bit)->width_contained(); ++bit) {}
      if (bit == end) break;

      int interval = (*bit)->x1() - (*ait)->x2();
      if(interval > wthres) {
        trlist->insert(it, new TextRect(head, ++ait));
        head = ait;
      }
    }

    if (head != cclist.begin()) {
      trlist->insert(it, new TextRect(head, end));

      TestUtils::Log("Split", (*it)->ToString());
      TestUtils::Log("CharNum", (*it)->CharNum());
      TestUtils::Log("median_interval", median_interval);
      TestUtils::Log("median_width", median_width);
      TestUtils::Log("wthres", wthres);

      it = trlist->erase(it);
    } else {
      ++it;
    }
  }

  it = trlist->begin();
  while (it != trlist->end()) {
    if ((*it)->CharNum() == 1) {
      it = trlist->erase(it);
      TestUtils::Log("Remove single char", (*it)->ToString());
    } else {
      ++it;
    }
  }
}

void TextDetector::SaveRegionTree(const string& dir_path, Region* root) {
	static int count = 0;

	mkdir(dir_path.c_str(), 0755);

	stack<Region*> regstack;
	stack<string> strstack;
	regstack.push(root);
	strstack.push(dir_path);
	while (!regstack.empty()) {
		Region* reg = regstack.top();
		regstack.pop();
		string sub_dir = strstack.top();
		strstack.pop();

		stringstream ss;
		ss << sub_dir << "/" << reg->level() << "_" << ++count;
		sub_dir = ss.str();

		Mat mask;
		reg->BuildMask(&mask);
		imwrite(sub_dir + ".bmp", mask);
		if (reg->IsLeaf()) continue;

		mkdir(sub_dir.c_str(), 0755);
		vector<Region*>::const_iterator it = reg->children().begin(), end = reg->children().end();
		for (int i = 0; it != end; ++it, ++i) {
			regstack.push(*it);
			strstack.push(sub_dir);
		}
	}
}
