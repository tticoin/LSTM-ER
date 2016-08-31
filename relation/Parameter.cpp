/*
 * Parameter.cpp
 *
 *  Created on: 2014/07/04
 *      Author: miwa
 */

#include "Parameter.h"
#include <fstream>
#include <cassert>
#include <cmath>
#include <boost/algorithm/string.hpp>
#include <unordered_set>
using std::unordered_set;

template<class T> T get(const YAML::Node& node, const string& s, T default_value=T()){
  if(!node[s] || node[s].IsNull()){
    return default_value;
  }else{
    return node[s].as<T>();
  }
}

namespace YAML{
template<> struct convert<coin::ParseParameters>{
  static Node encode(const coin::ParseParameters& params) {
    //TODO: encoding
    Node node;
    assert(false);
    return node;
  }
  static bool decode(const Node& node, coin::ParseParameters& params){
    assert(node.IsMap());
    params.extension_ = get<string>(node, "extension", ".split.stanford.so");
    params.sentence_tag_ = get<string>(node, "sentenceTag", "sentence");
    params.token_tag_ = get<string>(node, "tokenTag", "token");
    params.head_attr_ = get<string>(node, "headAttributeType", "");
    params.base_attr_ = get<string>(node, "baseAttributeType", "base");
    params.id_attr_ = get<string>(node, "idAttributeType", "id");
    params.pos_attr_ = get<string>(node, "posAttributeType", "pos");
    params.wn_attr_ = get<string>(node, "wordNetAttributeType", "WN");
    params.pred_attr_ = get<string>(node, "predicateAttributeType", "");
    const Node& attrs = node["tokenFeatureAttributeTypes"];
    for(unsigned i = 0;i < attrs.size();++i){
      params.token_attrs_.push_back(attrs[i].as<string>());
    }
    return true;
  }
};
}

namespace coin {

int VERBOSITY;


Parameters::Parameters(const string& file) {
  try{
    const YAML::Node& doc = YAML::LoadFile(file);
    assert(doc.IsMap());
    VERBOSITY = get(doc, "verbosity", 3);
    test_iteration_ = get(doc, "testIteration", -1);
    word_dimension_ = get(doc, "wordDimension", 50);
    pos_dimension_ = get(doc, "posDimension", 8);
    wn_dimension_ = get(doc, "wordNetDimension", 0);
    dep_dimension_ = get(doc, "depDimension", 8);
    label_dimension_ = get(doc, "labelDimension", 8);
    h1dimension_ = get(doc, "h1dimension", 32);
    h2dimension_ = get(doc, "h2dimension", 32);
    h3dimension_ = get(doc, "h3dimension", 32);
    iteration_ = get(doc, "iteration", 20);
    entity_iteration_ = get(doc, "entityIteration", 20);
    min_frequency_ = get(doc, "minFrequency", 2);
    minibatch_= get(doc, "minibatch", 1);
    lstm_layers_ = get(doc, "lstmLayers", 1);
    train_beam_size_ = get(doc, "trainBeamSize", 1);
    decode_beam_size_ = get(doc, "decodeBeamSize", 1);
    idropout_ = get(doc, "idropout", 0.1);
    hdropout_ = get(doc, "hdropout", 0.1);
    h2dropout_ = get(doc, "h2dropout", 0.1);
    odropout_ = get(doc, "odropout", 0.1);
    unk_prob_ = get(doc, "unkProb", 0.1);
    lambda_ = get(doc, "lambda", 1e-5);
    epsilon_ = get(doc, "epsilon", 1e-6);
    rho_ = get(doc, "rho", 0.95);
    eta0_ = get(doc, "eta0", 0.1);
    eta_decay_ =  get(doc, "etaDecay", 0.05);
    forget_bias_ = get(doc, "forgetBias", -1.);
    scheduling_k_ = get(doc, "schedulingK", 1.0);
    gradient_scale_ = get(doc, "gradientScale", 1.0);
    clip_threshold_ = get(doc, "clipThreshold", 5.);
    clipping_enabled_ = get(doc, "clippingEnabled", true);
    do_ner_ = get(doc, "doNER", true);
    do_rel_ = get(doc, "doRE", true);
    stack_seq_ = get(doc, "stackSeq", false);
    stack_tree_ = get(doc, "stackTree", false);
    use_reverse_rel_ = get(doc, "useReverseRel", false);
    use_base_ = get(doc, "useBase", false);
    use_lowercase_ = get(doc, "useLowercase", true);
    use_pair_exp_ = get(doc, "usePairExp", false);
    use_rel_exp_ = get(doc, "useRelExp", false);
    use_sp_exp_ = get(doc, "useSpExp", false);
    use_sp_tree_exp_ = get(doc, "useSpTreeExp", false);
    use_sp_full_tree_exp_ = get(doc, "useSpFullTreeExp", false);
    use_sp_sub_tree_exp_ = get(doc, "useSpSubTreeExp", false);
    if(use_sp_sub_tree_exp_){
      use_sp_full_tree_exp_ = true;
    }
    assert(use_pair_exp_ || use_rel_exp_ ||  use_sp_exp_ ||
           use_sp_tree_exp_ || use_sp_full_tree_exp_);
    train_dir_ = get<string>(doc, "trainDirectory", "train/");
    test_dir_ = get<string>(doc, "testDirectory", "test/");
    
    text_ext_ = get<string>(doc, "textExtension", ".txt");
    ann_ext_ = get<string>(doc, "annotationExtension", ".ann");
    pred_ext_ = get<string>(doc, "predictionExtension", ".pred.ann");

    model_file_ = get<string>(doc, "modelFile", "model.bin");
    
    string w2vFile = get<string>(doc, "w2vFile", "");
    if(w2vFile != ""){
      vector<string> wvfiles;
      boost::algorithm::split(wvfiles, w2vFile, bind2nd(std::equal_to<char>(), ','));
      w2v_dimension_ = 0;
      vector<unordered_map<string, vector<float> > > w2vs;
      unordered_set<string> keys;
      for(string wvfile:wvfiles){
        unordered_map<string, vector<float> > w2v;
        w2v_dimension_ += read_w2v(wvfile, w2v);
        w2vs.push_back(w2v);
        if(keys.size() == 0){
          for(auto& wv:w2v){
            keys.insert(wv.first);
          }
        }else{
          for(unordered_set<string>::iterator it = keys.begin();
              it != keys.end();){
            if(w2v.find(*it) == w2v.end()){
              it = keys.erase(it);
            }else{
              ++it;
            }
          }
        }
      }
      for(string key:keys){
        vector<float> v;
        for(auto &w2v:w2vs){
          v.insert(v.end(), w2v[key].begin(), w2v[key].end());
        }
        assert(v.size() == w2v_dimension_);
        w2v_.insert(make_pair(key, v));
      }
    }
    string p2vFile = get<string>(doc, "p2vFile", "");
    if(p2vFile != ""){
      p2v_dimension_ = read_w2v(p2vFile, p2v_);
    }
    
    const YAML::Node& parses = doc["parseParameters"];
    for(YAML::const_iterator it = parses.begin();it != parses.end();++it){
      string type = it->first.as<string>();
      parsers_[type] = it->second.as<ParseParameters>();
      parsers_[type].set_type(type);
    }
    base_parser_ = get<string>(doc, "baseParser", "");
    parsers_[base_parser_].set_base();
  }catch(YAML::BadFile& ex){
    std::cerr << "Bad file:" << ex.what() << std::endl;
    exit(0);
  }catch(YAML::Exception& ex){
    std::cerr << "Exception:" << ex.what() << std::endl;
    exit(0);
  }
}

Parameters::~Parameters() {
}

int Parameters::read_w2v(const string& w2vfile, unordered_map<string, vector<float> >& w2v){
#include <cstdlib>
#include <cstdio>
  const long long max_w = 50;              // max length of vocabulary entries  
  FILE *f;
  f = fopen(w2vfile.c_str(), "rb");
  if(f == NULL){
    cerr << w2vfile << " not found" << endl;
    exit(-1);
  }
  long long words, size;
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  cerr << words << " words, " << size << " dimensions" << endl;
  char vocab[max_w];
  // double maxv = 0.;
  for (long long b = 0; b < words; b++) {
    long long a = 0;
    while (1) {
      vocab[a] = fgetc(f);
      if (a == 0 && vocab[a] == ' ') continue;
      if (feof(f) || (vocab[a] == ' ')) break;
      if ((a < max_w) && (vocab[a] != '\n')) a++;
    }
    vocab[a] = '\0';
    string word(vocab);
    vector<float> vec;
    for (long long a = 0; a < size; a++){
      float v;
      fread(&v, sizeof(float), 1, f);
      vec.push_back(v);
      // if(maxv < fabs(v)){
      //   maxv = fabs(v);
      // }
    }
    w2v.insert(make_pair(word, vec));
  }
  // double scale = 20.0 / maxv;
  // for(auto& p:w2v){
  //   for(float &v:p.second){
  //     v *= scale;
  //   }
  // }
  fclose(f);
  return size;
}


void print_mem(){
  std::ifstream proc_stream("/proc/self/statm");
  long long VmSize = 0, VmRSS = 0, Share = 0;
  proc_stream >> VmSize >> VmRSS >> Share;
  proc_stream.close();
  std::cerr << "Memory usage: " << VmRSS  * sysconf(_SC_PAGESIZE) / (1024.0 * 1024.0) << " MiB." << std::endl;
}

} /* namespace coin */
