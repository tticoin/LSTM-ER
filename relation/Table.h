/*
 * Table.h
 *
 *  Created on: 2014/07/10
 *      Author: miwa
 */
#ifndef TABLE_H_
#define TABLE_H_

#include <vector>
#include <deque>
#include <iostream>
#include <cassert>
#include <unordered_set>
#include "Document.h"
using std::vector;
using std::unordered_set;

namespace coin {

class Table;

class TableCell{
private:
  Table& table_;
  int row_; // to or arg2
  int col_; // from or arg1
  string gold_id_;
  int gold_label_;
  int pred_label_;
  Word* word_;
public:
  TableCell(Table& table, int row, int col):
    table_(table), row_(row), col_(col), gold_id_(""), gold_label_(-1), pred_label_(-1), word_(nullptr){
    if(row < 0 || col < 0){
      std::cerr << "wrong cell index at (" << row << "," << col << ")" << endl;
    }
    assert(row >= 0);
    assert(col >= 0);
  };
  TableCell(Table& table, int row, int col, Word *word):
    table_(table), row_(row), col_(col), gold_id_(""), gold_label_(word->label_id()), pred_label_(-1), word_(word){
    if(word->direct_term() != nullptr){
      gold_id_ = word->direct_term()->id();
    }
    if(row < 0 || col < 0){
      std::cerr << "wrong cell index at (" << row << "," << col << ")" << endl;
    }
    assert(row >= 0);
    assert(col >= 0);
  }
  void set_gold_id(const string &gold_id){
    gold_id_ = gold_id;
  }
  void set_gold_label(int gold_label) {
    gold_label_ = gold_label;
  }
  void set_pred_label(int pred_label) {
    pred_label_ = pred_label;
  }
  int col() const {
    return col_;
  }
  string gold_id() const {
    return gold_id_;
  }
  int gold_label() const {
    return gold_label_;
  }
  int pred_label() const {
    return pred_label_;
  }
  int row() const {
    return row_;
  }
  const Word *word() const{
    return word_;
  }
  Table& table(){
    return table_;
  }
  bool is_entity_correct(const Dictionary& dict) const;
};

class Sentence;

class Table {
private:
  int size_;
  int seq_size_;
  TableCell*** cells_;
  vector<TableCell*> *sequence_;
  const Sentence& sentence_;
public:
  Table(const Sentence& sentence);
  virtual ~Table();
  const vector<TableCell*>* sequence() const{
    return sequence_;
  }
  TableCell* cell(int row, int col){
    return cells_[row][col];
  }
  const vector<int>& chars(int i){
    return cell(i, i)->word()->char_ids();
  }
  int word(int i){
    return cell(i, i)->word()->repr_id();
  }
  int pos(int i){
    return cell(i, i)->word()->pos_id();
  }
  int wn(int i){
    return cell(i, i)->word()->wn_id();
  }
  int pred_label(int i){
    return cell(i, i)->pred_label();
  }
  int seq_size() const {
    return seq_size_;
  }
  int size() const {
    return size_;
  }
  deque<int> get_path(int from, int to) const{
    return sentence_.get_path(from, to);
  }
  const Sentence& sentence() const{
    return sentence_;
  }
};

} /* namespace coin */

#endif /* TABLE_H_ */
