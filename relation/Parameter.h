/*
 * Parameter.h
 *
 *  Created on: 2014/07/04
 *      Author: miwa
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <yaml-cpp/yaml.h>
#include <string>
#include <chrono>
#include <vector>
#include <stack>
#include <unordered_map>
#include <iostream>
#include <cassert>
using std::string;
using std::stack;
using std::vector;
using std::unordered_map;
using std::pair;
using std::cout;
using std::cerr;
using std::endl;

namespace coin {

const string REVERSE_DEP_HEADER = "<-";
const string REVERSE_RELATION_HEADER = "REV";
const string NEGATIVE_RELATION = "Arg1:Other:Arg2";
const int NEGATIVE_DEPENDENCY_ID = 0;
const int NEGATIVE_ENTITY_ID = 0;
const int NEGATIVE_RELATION_ID = 0;
const int NEGATIVE_WORDNET_ID = 0;
extern int VERBOSITY;

class Timer{
private:
  std::chrono::time_point<std::chrono::system_clock> start_;
public:
  Timer(): start_(std::chrono::system_clock::now()){}
  double seconds() const{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()-start_).count() / 1000.;
  }
};

extern void print_mem();

class ParseParameters {
private:
  bool is_base_;
  string type_;
  string extension_;
  string head_attr_;
  string base_attr_;
  string id_attr_;
  string pos_attr_;
  string wn_attr_;
  string pred_attr_;
  string sentence_tag_;
  string token_tag_;
  vector<string> token_attrs_;
public:
  ParseParameters():is_base_(false){}
  bool is_base() const {
    return is_base_;
  }
  void set_base(){
    is_base_ = true;
  }
  const string& type() const {
    return type_;
  }
  void set_type(const string& type) {
    type_ = type;
  }
  const string& base_attr() const {
    return base_attr_;
  }
  const string& extension() const {
    return extension_;
  }
  const string& head_attr() const {
    return head_attr_;
  }
  const string& id_attr() const {
    return id_attr_;
  }
  const string& pos_attr() const {
    return pos_attr_;
  }
  const string& wn_attr() const {
    return wn_attr_;
  }  
  const string& pred_attr() const {
    return pred_attr_;
  }
  const string& sentence_tag() const {
    return sentence_tag_;
  }
  const vector<string>& token_attrs() const {
    return token_attrs_;
  }
  const string& token_tag() const {
    return token_tag_;
  }
  friend struct YAML::convert<ParseParameters>;
};

class Parameters {
private:
  int word_dimension_;
  int pos_dimension_;
  int wn_dimension_;
  int dep_dimension_;
  int label_dimension_;
  int h1dimension_;
  int h2dimension_;
  int h3dimension_;
  int iteration_;
  int entity_iteration_;
  int min_frequency_;
  int min_word_frequency_;
  int min_dep_frequency_;
  int minibatch_;
  int lstm_layers_;
  int test_iteration_;
  size_t train_beam_size_;
  size_t decode_beam_size_;  
  double idropout_;
  double hdropout_;
  double h2dropout_;
  double odropout_;
  double unk_prob_;
  double unk_dep_prob_;
  double lambda_;
  double epsilon_;
  double rho_;
  double eta0_;
  double eta_decay_;
  double clip_threshold_;
  double forget_bias_;
  double scheduling_k_;
  double gradient_scale_;
  bool clipping_enabled_;
  bool do_ner_;
  bool do_rel_;
  bool stack_seq_;
  bool stack_tree_;
  bool use_reverse_rel_;
  bool use_base_;
  bool use_lowercase_;
  bool use_pair_exp_;
  bool use_rel_exp_;
  bool use_sp_exp_;
  bool use_sp_tree_exp_;
  bool use_sp_full_tree_exp_;
  bool use_sp_sub_tree_exp_;
  string train_dir_, test_dir_;
  string text_ext_, ann_ext_, pred_ext_;
  string model_file_;
  unordered_map<string, ParseParameters> parsers_;
  int w2v_dimension_ = 0;
  unordered_map<string, vector<float> > w2v_;
  int p2v_dimension_ = 0;
  unordered_map<string, vector<float> > p2v_;
  string base_parser_;
  int read_w2v(const string& w2vfile, unordered_map<string, vector<float> >& w2v);
public:
  Parameters(const string& file);
  virtual ~Parameters();

  const string& ann_ext() const {
    return ann_ext_;
  }
  const string& base_parser() const {
    return base_parser_;
  }
  const bool do_ner() const {
    return do_ner_;
  }
  const bool do_rel() const {
    return do_rel_;
  }
  const bool stack_seq() const {
    return stack_seq_;
  }
  const bool stack_tree() const {
    return stack_tree_;
  }
  const bool use_reverse_rel() const{
    return use_reverse_rel_;
  }
  const bool use_base() const {
    return use_base_;
  }
  const bool use_lowercase() const {
    return use_lowercase_;
  }
  int test_iteration() const{
    return test_iteration_;
  }
  int entity_iteration() const{
    return entity_iteration_;
  }
  int w2v_dimension() const{
    return w2v_dimension_;
  }
  int p2v_dimension() const{
    return p2v_dimension_;
  }
  int word_dimension() const {
    return word_dimension_;
  }
  int pos_dimension() const {
    return pos_dimension_;
  }
  int wn_dimension() const {
    return wn_dimension_;
  }
  int dep_dimension() const {
    return dep_dimension_;
  }
  int label_dimension() const {
    return label_dimension_;
  }
  int h1dimension() const {
    return h1dimension_;
  }
  int h2dimension() const {
    return h2dimension_;
  }
  int h3dimension() const {
    return h3dimension_;
  }
  int iteration() const {
    return iteration_;
  }
  int min_frequency() const {
    return min_frequency_;
  }
  int min_word_frequency() const {
    return min_word_frequency_;
  }
  int min_dep_frequency() const {
    return min_dep_frequency_;
  }
  int minibatch() const{
    return minibatch_;
  }
  double clip_threshold() const{
    return clip_threshold_;
  }
  double forget_bias() const{
    return forget_bias_;
  }
  int lstm_layers() const{
    return lstm_layers_;
  }
  size_t train_beam_size() const{
    return train_beam_size_;
  }
  size_t decode_beam_size() const{
    return decode_beam_size_;
  }
  double scheduling_k() const{
    return scheduling_k_;
  }
  double gradient_scale() const{
    return gradient_scale_;
  }
  double idropout() const{
    return idropout_;
  }
  double hdropout() const{
    return hdropout_;
  }
  double h2dropout() const{
    return h2dropout_;
  }
  double odropout() const{
    return odropout_;
  }
  double unk_prob() const{
    return unk_prob_;
  }
  double unk_dep_prob() const{
    return unk_dep_prob_;
  }
  bool use_pair_exp() const{
    return use_pair_exp_;
  }
  bool use_rel_exp() const{
    return use_rel_exp_;
  }
  bool use_sp_exp() const{
    return use_sp_exp_;
  }
  bool use_sp_tree_exp() const{
    return use_sp_tree_exp_;
  }
  bool use_sp_full_tree_exp() const{
    return use_sp_full_tree_exp_;
  }
  bool use_sp_sub_tree_exp() const{
    return use_sp_sub_tree_exp_;
  }
  double lambda() const{
    return lambda_;
  }
  double epsilon() const{
    return epsilon_;
  }
  double rho() const{
    return rho_;
  }
  double eta0() const{
    return eta0_;
  }    
  double eta_decay() const{
    return eta_decay_;
  }
  bool clipping_enabled() const{
    return clipping_enabled_;
  }
  const unordered_map<string, ParseParameters>& parsers() const {
    return parsers_;
  }
  const string& pred_ext() const {
    return pred_ext_;
  }
  const string& test_dir() const {
    return test_dir_;
  }
  const string& text_ext() const {
    return text_ext_;
  }
  const string& train_dir() const {
    return train_dir_;
  }
  const string& model_file() const {
    return model_file_;
  }
  const unordered_map<string, vector<float> >& w2v() const{
    return w2v_;
  }
  const unordered_map<string, vector<float> >& p2v() const{
    return p2v_;
  }
  
};

} /* namespace coin */

#endif /* RELATIONEMBEDDING_SRC_PARAMETERS_H_ */
