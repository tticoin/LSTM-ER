/*
 * Table.cpp
 *
 *  Created on: 2014/07/10
 *      Author: miwa
 */

#include "Table.h"
#include "Document.h"
#include <iostream>
#include <algorithm>
using namespace std;

namespace coin {

Table::Table(const Sentence& sentence):
        size_(sentence.words().size()), seq_size_(size_ * (size_ + 1) / 2),
        sequence_(new vector<TableCell*>()), sentence_(sentence){
  cells_ = new TableCell**[size_];
  const vector<Word*>& words = sentence.words();
  for(int i = 0; i < size_;++i){
    cells_[i] = new TableCell*[i+1];
    for(int j = 0;j < i+1;++j){
      if(i == j){
        cells_[i][j] = new TableCell(*this, i, j, words[i]);
      }else{
        cells_[i][j] = new TableCell(*this, i, j);
      }
    }
  }
  // relation labels
  for(IndexRelation rel:sentence.index_rels()){
    if(rel.arg2() == rel.arg1()){
      cerr << "self-relation is detected for " << rel.arg1() << endl;
    }
    if(cells_[rel.arg2()][rel.arg1()]->gold_label() >= 0){
      cerr << "nested relation is detected for (" << rel.arg1() << "," << rel.arg2() << ")" << endl;
    }
    assert(rel.arg2() != rel.arg1());
    assert(cells_[rel.arg2()][rel.arg1()]->gold_label() < 0);
    cells_[rel.arg2()][rel.arg1()]->set_gold_label(rel.type());
    cells_[rel.arg2()][rel.arg1()]->set_gold_id(rel.id());
  }
  for(int i = 0; i < size_;++i){
    for(int j = 0;j < i;++j){
      if(cells_[i][j]->gold_label() < 0){
        cells_[i][j]->set_gold_label(NEGATIVE_RELATION_ID);
      }
    }
  }
  //right to left, close first
  sequence_->reserve(seq_size_);
  for(int dist = 0; dist < (int)size_;++dist){
    for(int i = size_ - dist - 1;i >= 0;--i){
      if(i < 0 || i >= (int)size_ || i+dist < 0 || i+dist >= (int)size_){
        cerr << "out of sequence " << i << endl;
      }
      assert(i >= 0 && i < (int)size_);
      assert(i+dist >= 0 && i+dist < (int)size_);
      sequence_->push_back(cells_[i+dist][i]);
    }
  }
}

bool TableCell::is_entity_correct(const Dictionary& dict) const {
  if(col_ == row_){
    if(!dict.is_entity_label(pred_label_) || !dict.is_entity_label(gold_label_)){
      return false;
    }
    int start_label = dict.get_begin_label(pred_label_);
    int current = col_;
    while(current >= 0){
      if(table_.cell(current, current)->pred_label() != table_.cell(current, current)->gold_label()){
        return false;
      }
      if(table_.cell(current, current)->pred_label() == start_label){
        return true;
      }
      current--;
    }  
    return false;    
  }else{
    return table_.cell(row_, row_)->is_entity_correct(dict) &&
      table_.cell(col_, col_)->is_entity_correct(dict);
  }
}

Table::~Table() {
  for(int i = 0; i < size_;++i){
    for(int j = 0;j < i+1;++j){
      delete cells_[i][j];
    }
    delete[] cells_[i];
  }
  delete[] cells_;
  delete sequence_;
}

} /* namespace coin */
