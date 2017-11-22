/*
 * RelLstmModel.cpp
 *
 *  Created on: 2015/09/13 Author: miwa
 */

#include <algorithm>
#include <fstream>
#include <list>
#include "RelLstmModel.h"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/algorithm/string.hpp>
using namespace boost::algorithm;

using namespace std;

namespace coin {

RelLSTMModel::RelLSTMModel(const Parameters& params, Dictionary& dict):
  params_(params), dict_(dict), updates_(0), iterations_(0),
  word_dim_(params_.w2v_dimension() > 0 ? params_.w2v_dimension() : params_.word_dimension()),
  pos_dim_(params_.p2v_dimension() > 0 ? params_.p2v_dimension() : params_.pos_dimension()),
  seq_dim_(word_dim_ + pos_dim_ + params_.wn_dimension()),
  pair_dim_(((params_.stack_seq() || params_.use_pair_exp()) ? params_.h1dimension() * 2 : 0) +
            (params_.stack_tree() ? params_.h1dimension() * 2 : 0)),
  rel_dim_(((params_.stack_seq() || params_.stack_tree()) ?
            pair_dim_ : seq_dim_)
           + (params_.do_ner() ? params_.label_dimension() : 0)),
  // sgd_(new cnn::SimpleSGDTrainer(&model_, params_.lambda(), params_.eta0())),
  // sgd_(new cnn::MomentumSGDTrainer(&model_, params_.lambda(), params_.eta0(), params_.rho())),
  // sgd_(new cnn::AdagradTrainer(&model_, params_.lambda(), params_.eta0())),
  // sgd_(new cnn::AdadeltaTrainer(&model_, params_.lambda(), params_.epsilon(), params.rho())),
  sgd_(new cnn::AdamTrainer(&model_, params_.lambda(), params_.eta0())),
  // sgd_(new cnn::AdaMaxTrainer(&model_, params_.lambda(), params_.eta0())),
  // hidden layer
  eseqLstm_(params.lstm_layers(), seq_dim_, params_.h1dimension(), &model_, params_.forget_bias()),
  ieseqLstm_(params.lstm_layers(), seq_dim_, params_.h1dimension(), &model_, params_.forget_bias()),
  // fullTreeLstm_(1, params.lstm_layers(), seq_dim_ + params_.dep_dimension(), params_.h1dimension(), &model_, params_.forget_bias()),
  // ifullTreeLstm_(1, params.lstm_layers(), seq_dim_ + params_.dep_dimension(), params_.h1dimension(), &model_, params_.forget_bias()),
  depLstm_(params.lstm_layers(), rel_dim_ + params_.dep_dimension() * 2, params_.h2dimension(), &model_, params_.forget_bias()),
  idepLstm_(params.lstm_layers(), rel_dim_ + params_.dep_dimension() * 2, params_.h2dimension(), &model_, params_.forget_bias()),
  spDepLstm_(params.lstm_layers(), rel_dim_ + params_.dep_dimension(), params_.h2dimension(), &model_, params_.forget_bias()),
  ispDepLstm_(params.lstm_layers(), rel_dim_ + params_.dep_dimension(), params_.h2dimension(), &model_, params_.forget_bias()),
  subTreeLstm_(2, params.lstm_layers(), rel_dim_ + params_.dep_dimension(), params_.h2dimension(), &model_, params_.forget_bias()),
  isubTreeLstm_(2, params.lstm_layers(), rel_dim_ + params_.dep_dimension(), params_.h2dimension(), &model_, params_.forget_bias()),
  spTreeLstm_(1, params.lstm_layers(), rel_dim_ + params_.dep_dimension(), params_.h2dimension(), &model_, params_.forget_bias()),
  ispTreeLstm_(1, params.lstm_layers(), rel_dim_ + params_.dep_dimension(), params_.h2dimension(), &model_, params_.forget_bias()){
  // learning params
  sgd_->eta_decay = params_.eta_decay();
  sgd_->clipping_enabled = params_.clipping_enabled();
  sgd_->clip_threshold = params_.clip_threshold() * params_.minibatch();
  // dict (word, POS, dependency, label, wordnet embeddings)
  w2i = model_.add_lookup_parameters(dict_.repr_types(), cnn::Dim(word_dim_, 1));
  if(params_.w2v().size() > 0){
    read_w2v();
  }
  if(pos_dim_ > 0){
    p2i = model_.add_lookup_parameters(dict_.pos_types(), cnn::Dim(pos_dim_, 1));
    if(params_.p2v().size() > 0){
      read_p2v();
    }
  }
  if(params_.dep_dimension() > 0){
    d2i = model_.add_lookup_parameters(dict_.dep_types(), cnn::Dim(params_.dep_dimension(), 1));
  }
  if(params_.label_dimension() > 0){
    e2i = model_.add_lookup_parameters(dict_.ent_types(), cnn::Dim(params_.label_dimension(), 1));
  }
  if(params_.wn_dimension() > 0){
    wn2i = model_.add_lookup_parameters(dict_.wn_types(), cnn::Dim(params_.wn_dimension(), 1));
    // zero vector for WordNet '0' category 
    vector<float> v(params_.wn_dimension(), 0.f);
    wn2i->Initialize(NEGATIVE_WORDNET_ID, v);
  }
  // entity
  s2h = model_.add_parameters(cnn::Dim(params_.h3dimension(), pair_dim_ + params_.label_dimension()));
  s2h_bias = model_.add_parameters(cnn::Dim(params_.h3dimension(), 1));
  h2e = model_.add_parameters(cnn::Dim(dict_.ent_types(), params_.h3dimension()));
  h2e_bias = model_.add_parameters(cnn::Dim(dict_.ent_types(), 1));
  s2h_bias->lambda = 0.;
  h2e_bias->lambda = 0.;
  // rel
  p2h = model_.add_parameters(cnn::Dim(params_.h3dimension(), pair_dim_ * 2));
  b2h = model_.add_parameters(cnn::Dim(params_.h3dimension(), params_.h2dimension() * 2));
  d2h = model_.add_parameters(cnn::Dim(params_.h3dimension(), params_.h2dimension() * 2));
  t2h = model_.add_parameters(cnn::Dim(params_.h3dimension(), params_.h2dimension() * 3));
  f2h = model_.add_parameters(cnn::Dim(params_.h3dimension(), params_.h2dimension() * 3));
  h_bias = model_.add_parameters(cnn::Dim(params_.h3dimension(), 1));
  h2r = model_.add_parameters(cnn::Dim(dict_.rel_types(), params_.h3dimension()));
  h2r_bias = model_.add_parameters(cnn::Dim(dict_.rel_types(), 1));
  h_bias->lambda = 0.;
  h2r_bias->lambda = 0.;
  // reference scores
  e2s = model_.add_lookup_parameters(dict_.ent_types(), cnn::Dim(dict_.ent_types(), 1), false);
  r2s = model_.add_lookup_parameters(dict_.rel_types(), cnn::Dim(dict_.rel_types(), 1), false);
  
  init_entity_scores();
  init_relation_scores();
}

// entity label scores 
void RelLSTMModel::init_entity_scores(){
  int ntypes = dict_.ent_types();
  for(int i = 0;i < ntypes;++i){
    vector<float> v(ntypes, 0.f);    
    string t_i = dict_.get_ent_string(i);
    if(i == 0){
      v[0] = 1.;
    }else{
      for(int j = 1;j < ntypes;++j){
        string t_j = dict_.get_ent_string(j);
        if(i == j){
          v[j] = 1.;
        }else{
          v[j] = 0.;
        }
      }
    }
    e2s->Initialize(i, v);
  }
}

// rel label scores 
void RelLSTMModel::init_relation_scores(){
  int ntypes = dict_.rel_types();
  for(int i = 0;i < ntypes;++i){
    vector<float> v(ntypes, 0);
    if(i == 0){
      v[0] = 1.;
    }else{
      for(int j = 1;j < ntypes;++j){
        if(i == j){
          v[j] = 1.;
        }else{
          v[j] = 0.;
        }
      }
    }
    r2s->Initialize(i, v);
  }
}

// Load model
RelLSTMModel::RelLSTMModel(const Parameters& params, Dictionary& dict, ifstream &is):
  params_(params), dict_(dict), updates_(0), iterations_(0),
  word_dim_(params_.w2v_dimension() > 0 ? params_.w2v_dimension() : params_.word_dimension()),
  pos_dim_(params_.p2v_dimension() > 0 ? params_.p2v_dimension() : params_.pos_dimension()),
  seq_dim_(word_dim_ + pos_dim_ + params_.wn_dimension()),
  pair_dim_(((params_.stack_seq() || params_.use_pair_exp()) ? params_.h1dimension() * 2 : 0) +
            (params_.stack_tree() ? params_.h1dimension() * 2 : 0)),
  rel_dim_(((params_.stack_seq() || params_.stack_tree()) ?
            pair_dim_ : seq_dim_)
           + (params_.do_ner() ? params_.label_dimension() : 0)),
  // sgd_(new cnn::SimpleSGDTrainer(&model_, params_.lambda(), params_.eta0())),
  // sgd_(new cnn::MomentumSGDTrainer(&model_, params_.lambda(), params_.eta0(), params_.rho())),
  // sgd_(new cnn::AdagradTrainer(&model_, params_.lambda(), params_.eta0())),
  // sgd_(new cnn::AdadeltaTrainer(&model_, params_.lambda(), params_.epsilon(), params.rho())),
  sgd_(new cnn::AdamTrainer(&model_, params_.lambda(), params_.eta0())),
  // sgd_(new cnn::AdaMaxTrainer(&model_, params_.lambda(), params_.eta0())),
  // hidden layer
  eseqLstm_(params.lstm_layers(), seq_dim_, params_.h1dimension(), &model_, params_.forget_bias()),
  ieseqLstm_(params.lstm_layers(), seq_dim_, params_.h1dimension(), &model_, params_.forget_bias()),
  // fullTreeLstm_(1, params.lstm_layers(), seq_dim_ + params_.dep_dimension(), params_.h1dimension(), &model_, params_.forget_bias()),
  // ifullTreeLstm_(1, params.lstm_layers(), seq_dim_ + params_.dep_dimension(), params_.h1dimension(), &model_, params_.forget_bias()),
  depLstm_(params.lstm_layers(), rel_dim_ + params_.dep_dimension() * 2, params_.h2dimension(), &model_, params_.forget_bias()),
  idepLstm_(params.lstm_layers(), rel_dim_ + params_.dep_dimension() * 2, params_.h2dimension(), &model_, params_.forget_bias()),
  spDepLstm_(params.lstm_layers(), rel_dim_ + params_.dep_dimension(), params_.h2dimension(), &model_, params_.forget_bias()),
  ispDepLstm_(params.lstm_layers(), rel_dim_ + params_.dep_dimension(), params_.h2dimension(), &model_, params_.forget_bias()),
  subTreeLstm_(2, params.lstm_layers(), rel_dim_ + params_.dep_dimension(), params_.h2dimension(), &model_, params_.forget_bias()),
  isubTreeLstm_(2, params.lstm_layers(), rel_dim_ + params_.dep_dimension(), params_.h2dimension(), &model_, params_.forget_bias()),
  spTreeLstm_(1, params.lstm_layers(), rel_dim_ + params_.dep_dimension(), params_.h2dimension(), &model_, params_.forget_bias()),
  ispTreeLstm_(1, params.lstm_layers(), rel_dim_ + params_.dep_dimension(), params_.h2dimension(), &model_, params_.forget_bias()){
  sgd_->eta_decay = params.eta_decay();
  sgd_->clipping_enabled = params.clipping_enabled();
  sgd_->clip_threshold = params.clip_threshold();
  boost::archive::binary_iarchive ia(is);
  // dict
  ia >> dict_;
  w2i = model_.add_lookup_parameters(dict_.repr_types(), cnn::Dim(word_dim_, 1));
  if(pos_dim_ > 0){
    p2i = model_.add_lookup_parameters(dict_.pos_types(), cnn::Dim(pos_dim_, 1));
  }
  if(params_.dep_dimension() > 0){
    d2i = model_.add_lookup_parameters(dict_.dep_types(), cnn::Dim(params_.dep_dimension(), 1));
  }
  if(params_.label_dimension() > 0){
    e2i = model_.add_lookup_parameters(dict_.ent_types(), cnn::Dim(params_.label_dimension(), 1));
  }
  if(params_.wn_dimension() > 0){
    wn2i = model_.add_lookup_parameters(dict_.wn_types(), cnn::Dim(params_.wn_dimension(), 1));
  }
  // entity
  s2h = model_.add_parameters(cnn::Dim(params_.h3dimension(), pair_dim_ + params_.label_dimension()));
  s2h_bias = model_.add_parameters(cnn::Dim(params_.h3dimension(), 1));
  h2e = model_.add_parameters(cnn::Dim(dict_.ent_types(), params_.h3dimension()));
  h2e_bias = model_.add_parameters(cnn::Dim(dict_.ent_types(), 1));
  // rel
  p2h = model_.add_parameters(cnn::Dim(params_.h3dimension(), pair_dim_ * 2));
  b2h = model_.add_parameters(cnn::Dim(params_.h3dimension(), params_.h2dimension() * 2));
  d2h = model_.add_parameters(cnn::Dim(params_.h3dimension(), params_.h2dimension() * 2));
  t2h = model_.add_parameters(cnn::Dim(params_.h3dimension(), params_.h2dimension() * 3));
  f2h = model_.add_parameters(cnn::Dim(params_.h3dimension(), params_.h2dimension() * 3));
  h_bias = model_.add_parameters(cnn::Dim(params_.h3dimension(), 1));
  h2r = model_.add_parameters(cnn::Dim(dict_.rel_types(), params_.h3dimension()));
  h2r_bias = model_.add_parameters(cnn::Dim(dict_.rel_types(), 1));
  // reference scores
  e2s = model_.add_lookup_parameters(dict_.ent_types(), cnn::Dim(dict_.ent_types(), 1), false);
  r2s = model_.add_lookup_parameters(dict_.rel_types(), cnn::Dim(dict_.rel_types(), 1), false);

  ia >> model_;
}

void RelLSTMModel::save_model(ofstream &os){
  boost::archive::binary_oarchive oa(os);
  oa << (const Dictionary&)dict_;
  oa << (const cnn::Model&)model_;
}

void RelLSTMModel::read_w2v(){
  for(const auto& wv:params_.w2v()){
    int id = dict_.get_repr_id(wv.first);
    if(id != -1){
      w2i->Initialize(id, wv.second);
    }
  }
}

void RelLSTMModel::read_p2v(){
  for(const auto& wv:params_.p2v()){
    int id = dict_.get_pos_id(wv.first);
    if(id != -1){
      p2i->Initialize(id, wv.second);
    }
  }
}

void RelLSTMModel::init_params(cnn::ComputationGraph &cg){
  param_vars_.clear();
  // acquire paramters
  param_vars_.push_back(parameter(cg, s2h));
  param_vars_.push_back(parameter(cg, s2h_bias));
  param_vars_.push_back(parameter(cg, h2e));
  param_vars_.push_back(parameter(cg, h2e_bias));
  param_vars_.push_back(parameter(cg, p2h));
  param_vars_.push_back(parameter(cg, b2h));
  param_vars_.push_back(parameter(cg, d2h));
  param_vars_.push_back(parameter(cg, t2h));
  param_vars_.push_back(parameter(cg, f2h));
  param_vars_.push_back(parameter(cg, h_bias));
  param_vars_.push_back(parameter(cg, h2r));
  param_vars_.push_back(parameter(cg, h2r_bias));
}

void RelLSTMModel::predict(vector<Table*> &tables, bool do_update, bool output, bool ent_only){
  Timer t;
  cnn::rndeng->seed(iterations_);
  do_update_ = do_update;
  cnn::cnn_do_update = do_update;
  if(do_update_){
    if(iterations_ > 0 && iterations_ % params_.minibatch() == 0){
      sgd_->update_epoch();
    }
    iterations_++;
  }
  
  // scheduled sampling
  double k = params_.scheduling_k();
  double scheduled_threshold = k < 1.0 ? -0.0 : k / (k + exp(sgd_->epoch / k));
  std::uniform_real_distribution<double> distrib(0.0, 1.0);

  double loss = 0.;
  double ignored_ent = 0., ignored_rel = 0.;
  double etp = 0., efp = 0., efn = 0.;
  double rtp = 0., rfp = 0., rfn = 0.;
  double covered = 0., all = 0.;
  unordered_map<int, int> rtps, rfps, rfns;
  for(int i = 0;i < dict_.rel_types();++i){
    rtps[i] = 0;
    rfps[i] = 0;
    rfns[i] = 0;
  }
  
  int ntables = tables.size();
  vector<int> order(ntables);
  for (int t = 0; t < ntables; ++t){
    order[t] = t;
  }
  // shuffle
  if(do_update_){
    shuffle(order.begin(), order.end(), *cnn::rndeng);
  }

  int ent_id = 1;
  int rel_id = 1;
  int instances = 0;
  
  // init computation graph
  cnn::ComputationGraph *cg = nullptr;
  for(int t = 0; t < ntables; ++t){
    Table* table = tables[order[t]];
    if(do_update_){
      if(t % params_.minibatch() == 0){
        if(cg != nullptr && instances > 0){
          cg->backward();
          sgd_->update(params_.gradient_scale()/params_.minibatch());
          ++updates_;
          delete cg;
          cg = nullptr;
          instances = 0;;
        }
        if(cg == nullptr){
          cg = new cnn::ComputationGraph();
          init_params(*cg);
        }
      }
    }else{
      if(cg != nullptr){
        delete cg;
      }
      cg = new cnn::ComputationGraph();
      init_params(*cg);
    }

    vector<Expression> errs;

    ofstream ofs;
    if(output){
      string fname = table->sentence().doc().id()+params_.pred_ext();
      ofs.open(fname.c_str(), ofstream::out | ofstream::app);
    }
    // entity
    int ent_num = table->size();
    vector<Expression> seqExps(ent_num);
    bool use_seq = params_.stack_seq() || params_.use_pair_exp(); 
    if(use_seq || params_.stack_tree()){
      vector<pair<Expression, Expression> > seqExpPairs(ent_num);
      vector<pair<Expression, Expression> > treeExpPairs(ent_num);
      if(use_seq || params_.use_pair_exp()){
        calc_seq_expressions(*cg, table, seqExpPairs);
      }
      if(params_.stack_tree()){
        calc_fulltree_expressions(*cg, table, treeExpPairs);
      }
      if(use_seq && params_.stack_tree()){
        for(int i = 0;i < ent_num;++i){
          seqExps[i] = concatenate({seqExpPairs[i].first, seqExpPairs[i].second,
                treeExpPairs[i].first, treeExpPairs[i].second});
        }
      }else if(use_seq){
        for(int i = 0;i < ent_num;++i){
          seqExps[i] = concatenate({seqExpPairs[i].first, seqExpPairs[i].second});
        }
      }else{
        for(int i = 0;i < ent_num;++i){
          seqExps[i] = concatenate({treeExpPairs[i].first, treeExpPairs[i].second});
        }
      }
    }
    int prev = NEGATIVE_ENTITY_ID;
    for(int i = 0;i < ent_num;++i){
      TableCell *cell = table->cell(i, i);
      if(params_.do_ner()){
        Expression entExp;
        if(params_.label_dimension() > 0){
          entExp = dropout_output(concatenate({seqExps[i], lookup(*cg, e2i, prev)}), params_.h2dropout());
        }else{
          entExp = dropout_output(seqExps[i], params_.h2dropout());
        }
        Expression h = dropout_output(tanh(affine_transform({param_vars_[SBH], param_vars_[S2H], entExp})), params_.odropout());
        Expression f = affine_transform({param_vars_[BE], param_vars_[H2E], h});
        vector<float> dist = as_vector(cg->incremental_forward());
        double best = -9e99;
        int besti = -1;
        bool is_last = (i == (ent_num - 1));
        const unordered_set<int>& nexts = dict_.next_entities(prev, is_last);
        for (int i:nexts){
          if (dist[i] > best) { best = dist[i]; besti = i; }
        }
        if(besti == -1 || cell->gold_label() < 0){
          cerr << "no prediction/no gold" << endl;
        }
        assert(besti >= 0);
        assert(cell->gold_label() >= 0);
        if(do_update){
          if(distrib(*cnn::rndeng) < scheduled_threshold &&
             nexts.find(cell->gold_label()) != nexts.end()){
            cell->set_pred_label(cell->gold_label());
          }else{
            cell->set_pred_label(besti);
          }
        }else{
          cell->set_pred_label(besti);
        }
        if (cell->is_entity_correct(dict_)) {
          etp++;
        }else{
          if(dict_.is_entity_label(besti)){
            efp++;
          }
          if(dict_.is_entity_label(cell->gold_label())){
            efn++;
          }
        }       
        prev = cell->pred_label();
        errs.push_back(cross_entropy_loss(f, const_lookup(*cg, e2s, cell->gold_label())));
        //errs.push_back(pickneglogsoftmax(f, cell->gold_label()));
        instances++;
      }else{
        cell->set_pred_label(cell->gold_label());
      }
    }
    map<int, int> ent_map;
    if(output && params_.do_ner()){
      list<pair<int, int> > pred_entities;
      for(int i = ent_num-1;i >= 0;--i){
        TableCell *cell = table->cell(i, i);
        if(dict_.is_entity_label(cell->pred_label())){
          int end = i;
          int start = end;
          int start_label = dict_.get_begin_label(cell->pred_label()); 
          while(start >= 0 && table->cell(start, start)->pred_label() != start_label){
            --start;
          }
          i = start;
          pred_entities.push_front(make_pair(start, end));
        }
      }
      for(pair<int, int> pred_entity:pred_entities){
        int start = pred_entity.first;
        int end = pred_entity.second;
        int start_offset = table->cell(start, start)->word()->start();
        int end_offset = table->cell(end, end)->word()->end();
        if(!params_.do_ner()){
          ent_id = atoi(table->cell(end, end)->gold_id().substr(1).c_str());
        }
        ofs << "T" << ent_id << "\t";
        ofs << dict_.get_ent_string(table->cell(end, end)->pred_label()).substr(2) << " ";
        ofs << start_offset << " ";
        ofs << end_offset << "\t";
        ofs << table->sentence().doc().text(start_offset, end_offset - start_offset) << endl;
        ent_map[end] = ent_id;
        ++ent_id;
      }
    }    
    if(!ent_only && params_.do_rel()){
      // relation
      vector<pair<Expression, Expression> > treeExps(ent_num);
      for(TableCell *cell:*(table->sequence())){
        if(cell->row() == cell->col())continue;
        bool pred =
          dict_.is_entity_label(table->cell(cell->row(), cell->row())->pred_label()) &&
          dict_.is_entity_label(table->cell(cell->col(), cell->col())->pred_label());
        if(pred){
          vector<Expression> rels;
          rels.push_back(param_vars_[BH]);
          if(params_.use_pair_exp()){
            rels.push_back(param_vars_[P2H]);
            rels.push_back(dropout_output(calc_pair_expression(*cg, cell, seqExps), params_.h2dropout()));
          }
          if(params_.use_rel_exp()){
            rels.push_back(param_vars_[B2H]);
            rels.push_back(dropout_output(calc_rel_expression(*cg, cell, seqExps), params_.h2dropout()));
          }
          if(params_.use_sp_exp()){
            rels.push_back(param_vars_[D2H]);
            rels.push_back(dropout_output(calc_sp_expression(*cg, cell, seqExps), params_.h2dropout()));
          }
          if(params_.use_sp_tree_exp()){
            rels.push_back(param_vars_[T2H]);
            rels.push_back(dropout_output(calc_sp_tree_expression(*cg, cell, seqExps), params_.h2dropout()));
          }
          if(params_.use_sp_full_tree_exp()){
            calc_subtree_expressions(*cg, cell, seqExps, treeExps);
            rels.push_back(param_vars_[F2H]);
            rels.push_back(dropout_output(calc_sp_subtree_expression(*cg, cell, treeExps), params_.h2dropout()));
          }
          Expression h = dropout_output(tanh(affine_transform(rels)), params_.odropout());
          Expression f = affine_transform({param_vars_[BR], param_vars_[H2R], h});
          vector<float> dist = as_vector(cg->incremental_forward());
          double best = -9e99;
          int besti = 0;
          int start = 0;
          for (int i = start; i < (int)dist.size(); ++i) {
            if (dist[i] > best) { best = dist[i]; besti = i; }
          }
          int forward_label = cell->gold_label();
          //if(params_.use_reverse_rel() && dict_.is_reverse_relation(forward_label)){
          //  forward_label = NEGATIVE_RELATION_ID;
          //}
          errs.push_back(cross_entropy_loss(f, const_lookup(*cg, r2s, forward_label)));
          //errs.push_back(pickneglogsoftmax(f, forward_label));
          instances++;
          if(params_.use_reverse_rel()){// right to left
            vector<Expression> r2l_rels;
            r2l_rels.push_back(param_vars_[BH]);
            if(params_.use_pair_exp()){
              r2l_rels.push_back(param_vars_[P2H]);
              r2l_rels.push_back(dropout_output(calc_pair_expression(*cg, cell, seqExps, true), params_.h2dropout()));
            }
            if(params_.use_rel_exp()){
              r2l_rels.push_back(param_vars_[B2H]);
              r2l_rels.push_back(dropout_output(calc_rel_expression(*cg, cell, seqExps, true), params_.h2dropout()));
            }
            if(params_.use_sp_exp()){
              r2l_rels.push_back(param_vars_[D2H]);
              r2l_rels.push_back(dropout_output(calc_sp_expression(*cg, cell, seqExps, true), params_.h2dropout()));
            }
            if(params_.use_sp_tree_exp()){
              r2l_rels.push_back(param_vars_[T2H]);
              r2l_rels.push_back(dropout_output(calc_sp_tree_expression(*cg, cell, seqExps, true), params_.h2dropout()));
            }
            if(params_.use_sp_full_tree_exp()){
              r2l_rels.push_back(param_vars_[F2H]);
              r2l_rels.push_back(dropout_output(calc_sp_subtree_expression(*cg, cell, treeExps, true), params_.h2dropout()));
            }
            Expression h = dropout_output(tanh(affine_transform(r2l_rels)), params_.odropout());
            Expression f = affine_transform({param_vars_[BR], param_vars_[H2R], h});
            vector<float> rdist = as_vector(cg->incremental_forward());
            double rbest = -9e99;
            int rbesti = 0;
            int rstart = 0;
            for (int i = rstart; i < (int)rdist.size(); ++i) {
              if (rdist[i] > rbest) { rbest = rdist[i]; rbesti = i; }
            }
            int backward_label = dict_.reverse_relation(cell->gold_label());
            //if(dict_.is_reverse_relation(backward_label)){
            //  backward_label = NEGATIVE_RELATION_ID;
            //}
            errs.push_back(cross_entropy_loss(f, const_lookup(*cg, r2s, backward_label)));
            instances++;
            //errs.push_back(pickneglogsoftmax(f, backward_label));
            // prediction
            rbesti = dict_.reverse_relation(rbesti); // normalize to ltor
            if(besti != rbesti){
              if(dict_.is_negative_relation(besti)){
                // select positive
                besti = rbesti;
              }else if(!dict_.is_negative_relation(rbesti)){
                if(best < rbest){
                  // select confident
                  besti = rbesti;
                }
              }
            }
          }
          if(!params_.do_ner() && !cell->is_entity_correct(dict_)){
            cerr << "no specification for entity detection." << endl;
          }
          assert(params_.do_ner() || cell->is_entity_correct(dict_));
          bool correct = false;
          if (cell->gold_label() == besti && cell->is_entity_correct(dict_)){
            correct = true;
            if(besti > 0){
              rtp++;
              rtps[besti]++;
            }
          }else{
            if(besti > 0){
              rfp++;
              rfps[besti]++;
            }
            if(cell->gold_label() > 0){
              rfn++;
              rfns[cell->gold_label()]++;
            }
            
          }
          if(cell->gold_label() > 0){
            covered++;
            all++;
          }
          // output??
          if(!do_update && iterations_ == params_.test_iteration()){
            if(cell->gold_id() != ""){
              cout << table->sentence().doc().id() << "\t" << cell->gold_id() << "\t" << dict_.get_rel_string(cell->gold_label()) << "\t" << dict_.get_rel_string(besti) << "\t" << (correct ? "true" : "false") << endl;
            }else{
              cout << table->sentence().doc().id() << "\t" << cell->row() << "->" << cell->col() << "\t" << dict_.get_rel_string(cell->gold_label()) << "\t" << dict_.get_rel_string(besti) << "\t" << "false" << endl;
            }
          }
          
          if(output){
            if(besti > 0){
              ofs << "R" << rel_id << "\t";
              if(dict_.is_reverse_relation(besti)){
                int rbesti = dict_.reverse_relation(besti); // normalize
                string rel_type = dict_.get_rel_string(rbesti);
                vector<string> rels;
                split(rels, rel_type, bind2nd(equal_to<char>(), ':'));
                ofs << rels[1] << " ";
                if(ent_map.find(cell->row()) != ent_map.end()){
                  ofs << "Arg1:T" << ent_map[cell->row()] << " ";
                  ofs << "Arg2:T" << ent_map[cell->col()] << endl;
                }else{
                  ofs << "Arg1:" << table->cell(cell->row(), cell->row())->gold_id() << " ";
                  ofs << "Arg2:" << table->cell(cell->col(), cell->col())->gold_id() << endl;
                }
              }else{
                string rel_type = dict_.get_rel_string(besti);
                vector<string> rels;
                split(rels, rel_type, bind2nd(equal_to<char>(), ':'));
                ofs << rels[1] << " ";
                if(ent_map.find(cell->row()) != ent_map.end()){
                  ofs << "Arg1:T" << ent_map[cell->col()] << " ";
                  ofs << "Arg2:T" << ent_map[cell->row()] << endl;
                }else{
                  ofs << "Arg1:" << table->cell(cell->col(), cell->col())->gold_id() << " ";
                  ofs << "Arg2:" << table->cell(cell->row(), cell->row())->gold_id() << endl;
                }
              }
              ++rel_id;
            }
          }
          // cerr << "label:" << dict_.get_rel_string(cell->gold_label()) << endl;
          // cerr << "pred:"<< dict_.get_rel_string(besti) << endl;
          // errs.push_back(hinge(f, cell->gold_label(), 10.0f));
        }else if(cell->gold_label() > 0){
          rfn++;
          rfns[cell->gold_label()]++;
          all++;
          if(!do_update && iterations_ == params_.test_iteration()){
            cout << table->sentence().doc().id() << "\t" << cell->gold_id() << "\t" << dict_.get_rel_string(cell->gold_label()) << "\t" << dict_.get_rel_string(NEGATIVE_RELATION_ID) << "\t" << "false" << endl;
          }        
        }
      }
    }
    if(errs.size() > 0){
      sum(errs);
      loss += as_scalar(cg->incremental_forward());
    }
    // add ignored
    if(params_.do_ner()){
      ignored_ent += table->sentence().missing_terms();
    }
    if(!ent_only && params_.do_rel()){
      ignored_rel += table->sentence().missing_rels();
    }
    if(output){
      ofs.close();
    }
  }
  if(do_update_ && instances > 0){
    cg->backward();
    sgd_->update(params_.gradient_scale()/params_.minibatch());
    ++updates_;
    sgd_->status();
    cerr << "\n";
    instances = 0;
  }
  if(cg != nullptr){
    delete cg;
  }
  
  double ep = (etp+efp == 0.) ? 0. : etp/(etp+efp);
  double er = (etp+efn == 0.) ? 0. : etp/(etp+efn);
  double ef = (ep == 0. || er == 0.) ? 0. : 2 * ep * er / (ep + er);
  double rp = (rtp+rfp == 0.) ? 0. : rtp/(rtp+rfp);
  double rr = (rtp+rfn == 0.) ? 0. : rtp/(rtp+rfn);
  double rf = (rp == 0. || rr == 0.) ? 0. : 2 * rp * rr / (rp + rr);

  double rps = 0., rrs = 0., rfs = 0., count = 0.;
  for(int i = 1;i < dict_.rel_types();i+=2){
    rtps[i] += rtps[i+1];
    rfps[i] += rfps[i+1];
    rfns[i] += rfns[i+1];
    assert(dict_.get_rel_string(i+1).find(dict_.get_rel_string(i)) != string::npos);
    if(rfns[i]+rtps[i] == 0)continue;
    double p = (rtps[i]+rfps[i] == 0) ? 0. : (double)rtps[i]/(rtps[i]+rfps[i]);
    double r = (rtps[i]+rfns[i] == 0) ? 0. : (double)rtps[i]/(rtps[i]+rfns[i]);
    double f = (p == 0. || r == 0.) ? 0. : 2 * p * r / (p + r);
    rps += p; rrs += r; rfs += f; count++;
  }
  if(count != 0.){
    rps /= count;
    rrs/= count;
    rfs /= count;
  }
  if(!output){
    cerr << "loss=" << loss << ", ent P/R/F = " << ep << "/" << er << "/" << ef << ", rel micro P/R/F=" << rp << "/" << rr << "/" << rf <<  ", macro P/R/F=" << rps << "/" << rrs << "/" << rfs << ", coverage=" << covered << "/" << all << ", " << t.seconds() << " seconds." << endl;
  }
  efn += ignored_ent;
  rfn += ignored_rel;
  all += ignored_rel;
  ep = (etp+efp == 0.) ? 0. : etp/(etp+efp);
  er = (etp+efn == 0.) ? 0. : etp/(etp+efn);
  ef = (ep == 0. || er == 0.) ? 0. : 2 * ep * er / (ep + er);
  rp = (rtp+rfp == 0.) ? 0. : rtp/(rtp+rfp);
  rr = (rtp+rfn == 0.) ? 0. : rtp/(rtp+rfn);
  rf = (rp == 0. || rr == 0.) ? 0. : 2 * rp * rr / (rp + rr);
  if(!output){
    cerr << "Add ignored ent P/R/F = " << ep << "/" << er << "/" << ef << ", rel micro P/R/F=" << rp << "/" << rr << "/" << rf <<  ", coverage=" << covered << "/" << all << endl;
  }
}


Expression RelLSTMModel::get_expression(cnn::ComputationGraph& cg, Table *table, int i){
  vector<Expression> expressions;
  expressions.push_back(dropout_lookup(cg, w2i, table->word(i)));
  if(pos_dim_ > 0){
    expressions.push_back(dropout_lookup(cg, p2i, table->pos(i)));
  }
  if(params_.wn_dimension() > 0){
    if(table->wn(i) == NEGATIVE_WORDNET_ID){
      // ignore if no wordnet hypernym (concat zero vector)
      expressions.push_back(const_lookup(cg, wn2i, table->wn(i)));
    }else{
      expressions.push_back(dropout_lookup(cg, wn2i, table->wn(i)));
    }
  }
  return concatenate(expressions);
}


Expression RelLSTMModel::get_expression(cnn::ComputationGraph& cg, Table *table, int i, vector<Expression> &seqExps){
  Expression ex;
  if(params_.stack_seq() || params_.stack_tree()){
    ex = dropout(seqExps[i], params_.hdropout());
  }else{
    ex = get_expression(cg, table, i);
  }
  if(!params_.do_ner() || params_.label_dimension() == 0){
    return ex;
  }else{
    return concatenate({ex, dropout_lookup(cg, e2i, table->pred_label(i))});
  }      
}

Expression RelLSTMModel::calc_region_expression(cnn::ComputationGraph& cg, Table* table, vector<Expression>& seqExps, int start, int end){
  if(start >= end){
    vector<cnn::real> x_values(params_.h1dimension() * 2, 0.); 
    Expression x = input(cg, {params_.h1dimension() * 2}, &x_values);
    cg.incremental_forward();
    return x;
  }else{
    vector<Expression>  v;
    for(int i = start;i < end;++i){
      v.push_back(seqExps[i]);
    }
    Expression lv = average(v);
//    Expression lv;
//    lv = seqExps[start];
//    for(int i = start+1;i < end;++i){
//      lv = max(lv, seqExps[i]);
//    }
    return lv;
  }
}

Expression RelLSTMModel::calc_pair_expression(cnn::ComputationGraph& cg, TableCell* cell, vector<Expression>& seqExps, bool reverse){
  assert(cell->row() > cell->col());
  Table* table = &(cell->table());
  int le = cell->col();
  int ls = le;
  int ls_label = dict_.get_begin_label(table->cell(le, le)->pred_label());
  while(ls >= 0){
    if(table->cell(ls, ls)->pred_label() == ls_label)break;
    --ls;
  }
  int re = cell->row();
  int rs = re;
  int rs_label = dict_.get_begin_label(table->cell(re, re)->pred_label());
  while(rs >= 0){
    if(table->cell(rs, rs)->pred_label() == rs_label)break;
    --rs;
  }
  assert(ls >= 0 && rs >= 0);
  assert(ls <= le && rs <= re);
  // regions.push_back(calc_region_expression(cg, table, seqExps, 0, ls));
  Expression e1 = calc_region_expression(cg, table, seqExps, le, le+1);
  // regions.push_back(calc_region_expression(cg, table, seqExps, le+params.lstm_layers(), rs));
  Expression e2 = calc_region_expression(cg, table, seqExps, re, re+1);
  // regions.push_back(calc_region_expression(cg, table, seqExps, re+1, table->size()));
  Expression pair;
  if(reverse){
    pair = concatenate({e2, e1});
  }else{
    pair = concatenate({e1, e2});
  }
  // Expression pair = calc_region_expression(cg, table, seqExps, le, re+1);
  return pair;
}

Expression RelLSTMModel::calc_rel_expression(cnn::ComputationGraph& cg, TableCell* cell, vector<Expression> &seqExps, bool reverse){
  assert(cell->row() > cell->col());
  Table* table = &(cell->table());
  deque<int> sp, rev_sp;
  if(reverse){
    sp = table->get_path(cell->row(), cell->col());
    rev_sp = table->get_path(cell->col(), cell->row());
  }else{
    sp = table->get_path(cell->col(), cell->row());
    rev_sp = table->get_path(cell->row(), cell->col());
  }
  int sp_size = sp.size();
  int rev_sp_size = rev_sp.size();
  assert(sp_size == rev_sp_size);
  Expression depFwdMax, depFwdRoot, depRevRoot, depRevMax;
  if(sp_size != 0){
    assert(sp_size > 2);
    sp.push_front(NEGATIVE_DEPENDENCY_ID);
    sp.push_back(NEGATIVE_DEPENDENCY_ID);
    sp_size += 2;
    // find least common node
    int lcn = -1;
    for(int i = 1;i < sp_size;i += 2){
      if(i == 1){
        if(sp[i-1] % 2 == sp[i+1] % 2){
          lcn = i;
        }
      }else{
        if(sp[i-1] % 2 != sp[i+1] % 2){
          assert(lcn == -1);
          lcn = i;
        }
      }
    }
    assert(lcn > 0 && lcn < sp_size - 1);
    depLstm_.new_graph(cg);
    depLstm_.start_new_sequence();
    for(int i = 1;i < sp_size;i += 2){
      Expression depFwd;
      if(params_.dep_dimension() > 0){
        depFwd = depLstm_.add_input(concatenate({get_expression(cg, table, sp[i], seqExps), dropout_lookup(cg, d2i, sp[i-1]), dropout_lookup(cg, d2i, sp[i+1])}));
      }else{
        depFwd = depLstm_.add_input(get_expression(cg, table, sp[i], seqExps));
      }
      // if(i == 1){
      //   depFwdMax = depFwd;
      // }else{
      //   depFwdMax = max(depFwd, depFwdMax);
      // }
      if(i == lcn){
        depFwdRoot = depFwd;
      }
      depFwdMax = depFwd;
    }

    rev_sp.push_front(NEGATIVE_DEPENDENCY_ID);
    rev_sp.push_back(NEGATIVE_DEPENDENCY_ID);
    rev_sp_size += 2;
    // find least common node
    int rlcn = -1;
    for(int i = 1;i < rev_sp_size;i += 2){
      if(i == 1){
        if(rev_sp[i-1] % 2 == rev_sp[i+1] % 2){
          rlcn = i;
        }
      }else{
        if(rev_sp[i-1] % 2 != rev_sp[i+1] % 2){
          assert(rlcn == -1);
          rlcn = i;
        }
      }
    }
    idepLstm_.new_graph(cg);
    idepLstm_.start_new_sequence();
    for(int i = 1;i < rev_sp_size;i += 2){
      Expression depRev;
      if(params_.dep_dimension() > 0){
        depRev = idepLstm_.add_input(concatenate({get_expression(cg, table, rev_sp[i], seqExps), dropout_lookup(cg, d2i, rev_sp[i-1]), dropout_lookup(cg, d2i, rev_sp[i+1])}));
      }else{
        depRev = idepLstm_.add_input(get_expression(cg, table, rev_sp[i], seqExps));
      }
      // if(i == 1){
      //   depRevMax = depRev;
      // }else{
      //   depRevMax = max(depRev, depRevMax);
      // }
      if(i == rlcn){
        depRevRoot = depRev;
      }
      depRevMax = depRev;
    }
    Expression relExp = concatenate({depFwdMax, depRevMax});
    // vector<float> v = as_vector(relExp.value());
    // for(float f:v){
    //   assert(!isnan(f));
    // }
    return relExp;
  }else{
    vector<cnn::real> x_values(params_.h1dimension() * 2, 0.); 
    Expression x = input(cg, {params_.h1dimension() * 2}, &x_values);
    cg.incremental_forward();
    return x;
  }
}

Expression RelLSTMModel::calc_sp_tree_expression(cnn::ComputationGraph& cg, TableCell* cell, vector<Expression> &seqExps, bool reverse){
  assert(cell->row() > cell->col());
  Table* table = &(cell->table());
  deque<int> sp = table->get_path(cell->col(), cell->row());
  int sp_size = sp.size();
  unordered_map<int, Expression> buExpressions;
  unordered_map<int, Expression> tdExpressions;  
  if(sp_size > 0){
    assert(sp_size > 2);
    sp.push_front(NEGATIVE_DEPENDENCY_ID);
    sp.push_back(NEGATIVE_DEPENDENCY_ID);
    sp_size += 2;
    // find least common node
    int lcn = -1;
    for(int i = 1;i < sp_size;i += 2){
      if(i == 1){
        if(sp[i-1] % 2 == sp[i+1] % 2){
          lcn = i;
        }
      }else{
        if(sp[i-1] % 2 != sp[i+1] % 2){
          assert(lcn == -1);
          lcn = i;
        }
      }
    }
    sp[sp.size()-1] += 1; // to mimick negative
    assert(lcn > 0 && lcn < sp_size - 1);
    Expression root, left, right;
    {
      // bottom up
      spTreeLstm_.new_graph(cg);
      spTreeLstm_.start_new_sequence();
      int bunodes = 0;
      int bu_first = bunodes;
      for(int i = 1;i < lcn;i += 2){
        Expression input;
        if(params_.dep_dimension() > 0){
          input = concatenate({get_expression(cg, table, sp[i], seqExps), dropout_lookup(cg, d2i, sp[i+1])});
        }else{
          input = get_expression(cg, table, sp[i], seqExps);
        }
        vector<pair<int,int> > children;
        if(i > 1){
          assert(bunodes-1>=0);
          children.push_back(make_pair(0, bunodes-1));
          // children.push_back(make_pair((sp[i-1]+1)/2, bunodes-1));
        }
        buExpressions.insert(make_pair(sp[i], spTreeLstm_.add_input(children, input)));
        bunodes++;
      }
      int left_child = bunodes - 1;
      for(int i = sp_size - 2;i > lcn;i -= 2){
        Expression input;
        if(params_.dep_dimension() > 0){
          input = concatenate({get_expression(cg, table, sp[i], seqExps), dropout_lookup(cg, d2i, sp[i-1]-1)});
        }else{
          input = get_expression(cg, table, sp[i], seqExps);
        }
        vector<pair<int,int> > children;
        if(i < sp_size - 2){
          children.push_back(make_pair(0, bunodes-1));
          //children.push_back(make_pair(sp[i+1]/2, bunodes-1));
        }
        buExpressions.insert(make_pair(sp[i], spTreeLstm_.add_input(children, input)));
        bunodes++;
      }
      {
        int right_child = bunodes - 1;
        vector<pair<int, int> > children;
        if(left_child != bu_first - 1){
          children.push_back(make_pair(0, left_child));
          // children.push_back(make_pair((sp[lcn-1]+1)/2, left_child));
        }
        if(left_child != right_child){
          children.push_back(make_pair(0, right_child));
          // children.push_back(make_pair(sp[lcn+1]/2, right_child));
        }
        Expression input;
        if(params_.dep_dimension() > 0){
          input = concatenate({get_expression(cg, table, sp[lcn], seqExps), dropout_lookup(cg, d2i, NEGATIVE_DEPENDENCY_ID)});
        }else{
          input = get_expression(cg, table, sp[lcn], seqExps);
        }
        assert(children.size() > 0);
        root = spTreeLstm_.add_input(children, input);
        buExpressions.insert(make_pair(sp[lcn], root));
        bunodes++;
      }

      // vector<Expression> init;
      // init.insert(init.end(), spTreeLstm_.c.back().begin(), spTreeLstm_.c.back().end());
      // init.insert(init.end(), spTreeLstm_.h.back().begin(), spTreeLstm_.h.back().end());
      // top down
      int tdnodes = 0; //bunodes;
      int td_first = tdnodes;
      ispTreeLstm_.new_graph(cg);
      //ispTreeLstm_.start_new_sequence(init);
      ispTreeLstm_.start_new_sequence();
      Expression tdroot;
      {
        Expression input;
        if(params_.dep_dimension() > 0){
          input = concatenate({get_expression(cg, table, sp[lcn], seqExps), dropout_lookup(cg, d2i, NEGATIVE_DEPENDENCY_ID)});
        }else{
          input = get_expression(cg, table, sp[lcn], seqExps);
        }
        vector<pair<int, int> > children;
        // children.push_back(make_pair(0, tdnodes - 1));
        tdroot = ispTreeLstm_.add_input(children, input);
        tdExpressions.insert(make_pair(sp[lcn], tdroot));
        tdnodes++;
      }
      for(int i = lcn - 2;i >= 1;i -= 2){
        Expression input;
        if(params_.dep_dimension() > 0){
          input = concatenate({get_expression(cg, table, sp[i], seqExps), dropout_lookup(cg, d2i, sp[i+1])});
        }else{
          input = get_expression(cg, table, sp[i], seqExps);
        }
        vector<pair<int, int> > children;
        children.push_back(make_pair(0, tdnodes - 1));
        // children.push_back(make_pair((sp[i+1]+1)/2, tdnodes - 1));
        left = ispTreeLstm_.add_input(children, input);
        tdExpressions.insert(make_pair(sp[i], left));
        tdnodes++;
      }
      if(tdnodes == td_first + 1){
        left = tdroot;
      }
      int rstart = tdnodes;
      for(int i = lcn + 2;i <= sp_size - 2;i += 2){
        Expression input;
        if(params_.dep_dimension() > 0){
          input = concatenate({get_expression(cg, table, sp[i], seqExps), dropout_lookup(cg, d2i, sp[i-1]-1)});
        }else{
          input = get_expression(cg, table, sp[i], seqExps);
        }
        vector<pair<int, int> > children;
        if(i == lcn + 2){
          children.push_back(make_pair(0, td_first));
          // children.push_back(make_pair(sp[i-1]/2, td_first));
        }else{
          children.push_back(make_pair(0, tdnodes - 1));
          // children.push_back(make_pair(sp[i-1]/2, tdnodes - 1));
        }
        right = ispTreeLstm_.add_input(children, input);
        tdExpressions.insert(make_pair(sp[i], right));
        tdnodes++;
      }
      if(rstart == tdnodes){
        right = tdroot;
      }
    }
    // {
    //   ispTreeLstm_.new_graph(cg);
    //   ispTreeLstm_.start_new_sequence();
    //   // top down
    //   int tdnodes = 0;
    //   int td_first = tdnodes;
    //   int tdleft, tdright;
    //   Expression tdroot;
    //   {
    //     Expression input = get_expression(cg, table, sp[lcn], seqExps);
    //     vector<pair<int, int> > children;
    //     tdroot = ispTreeLstm_.add_input(children, input);
    //     tdnodes++;
    //   }
    //   for(int i = lcn - 2;i >= 1;i -= 2){
    //     Expression input = get_expression(cg, table, sp[i], seqExps);
    //     vector<pair<int, int> > children;
    //     children.push_back(make_pair(0, tdnodes - 1));
    //     // children.push_back(make_pair((sp[i+1]+1)/2, tdnodes - 1));
    //     ispTreeLstm_.add_input(children, input);
    //     tdleft = tdnodes;
    //     tdnodes++;
    //   }
    //   if(tdnodes == td_first + 1){
    //     tdleft = td_first;
    //   }
    //   int rstart = tdnodes;
    //   for(int i = lcn + 2;i <= sp_size - 2;i += 2){
    //     Expression input = get_expression(cg, table, sp[i], seqExps);
    //     vector<pair<int, int> > children;
    //     if(i == lcn + 2){
    //       children.push_back(make_pair(0, td_first));
    //       // children.push_back(make_pair(sp[i-1]/2, td_first));
    //     }else{
    //       children.push_back(make_pair(0, tdnodes - 1));
    //       //children.push_back(make_pair(sp[i-1]/2, tdnodes - 1));
    //     }
    //     ispTreeLstm_.add_input(children, input);
    //     tdright = tdnodes;
    //     tdnodes++;
    //   }
    //   if(rstart == tdnodes){
    //     tdright = td_first;
    //   }
    //   // bottom up
    //   int bunodes = tdnodes;
    //   int bu_first = bunodes;
    //   for(int i = 1;i < lcn;i += 2){
    //     Expression input = get_expression(cg, table, sp[i], seqExps);
    //     vector<pair<int,int> > children;
    //     if(i > 1){
    //       assert(bunodes-1>=0);
    //       children.push_back(make_pair(0, bunodes-1));
    //       // children.push_back(make_pair((sp[i-1]+1)/2, bunodes-1));
    //     }else{
    //       children.push_back(make_pair(0, tdleft));
    //     }
    //     ispTreeLstm_.add_input(children, input);
    //     bunodes++;
    //   }
    //   int left_child = bunodes - 1;
    //   for(int i = sp_size - 2;i > lcn;i -= 2){
    //     Expression input = get_expression(cg, table, sp[i], seqExps);
    //     vector<pair<int,int> > children;
    //     if(i < sp_size - 2){
    //       children.push_back(make_pair(0, bunodes-1));
    //       //children.push_back(make_pair(sp[i+1]/2, bunodes-1));
    //     }else{
    //       children.push_back(make_pair(0, tdright));
    //     }
    //     ispTreeLstm_.add_input(children, input);
    //     bunodes++;
    //   }
    //   {
    //     int right_child = bunodes - 1;
    //     vector<pair<int, int> > children;
    //     if(left_child != bu_first - 1){
    //       children.push_back(make_pair(0, left_child));
    //       // children.push_back(make_pair((sp[lcn-1]+1)/2, left_child));
    //     }
    //     if(left_child != right_child){
    //       children.push_back(make_pair(0, right_child));
    //       // children.push_back(make_pair(sp[lcn+1]/2, right_child));
    //     }
    //     Expression input = get_expression(cg, table, sp[lcn], seqExps);
    //     assert(children.size() > 0);
    //     root = ispTreeLstm_.add_input(children, input);
    //     bunodes++;
    //   }
    // }
    vector<pair<Expression, Expression> > treeExps(seqExps.size());
    for(int i = 1;i < sp_size;i+=2){
      assert(buExpressions.find(sp[i]) != buExpressions.end());
      assert(tdExpressions.find(sp[i]) != tdExpressions.end());
      treeExps[sp[i]] = make_pair(buExpressions[sp[i]], tdExpressions[sp[i]]);
    }
    Expression x;
    if(reverse){
      x = concatenate({right, root, left});
    }else{
      x = concatenate({left, root, right});
    }
    return x;    
  }else{
    vector<cnn::real> x_values(params_.h1dimension() * 3, 0.); 
    Expression x = input(cg, {params_.h1dimension() * 3}, &x_values);
    cg.incremental_forward();
    return x;
  }
}

Expression RelLSTMModel::calc_sp_subtree_expression(cnn::ComputationGraph& cg, TableCell* cell, vector<pair<Expression, Expression> >& treeExps, bool reverse){
  Table* table = &(cell->table());
  deque<int> sp = table->get_path(cell->col(), cell->row());
  int sp_size = sp.size();
  if(sp_size > 0){
    assert(sp_size > 2);
    sp.push_front(NEGATIVE_DEPENDENCY_ID);
    sp.push_back(NEGATIVE_DEPENDENCY_ID);
    sp_size += 2;
    // find least common node
    int lcn = -1;
    for(int i = 1;i < sp_size;i += 2){
      if(i == 1){
        if(sp[i-1] % 2 == sp[i+1] % 2){
          lcn = i;
        }
      }else{
        if(sp[i-1] % 2 != sp[i+1] % 2){
          assert(lcn == -1);
          lcn = i;
        }
      }
    }
    assert(lcn > 0 && lcn < sp_size - 1);
    Expression x;
    if(reverse){
      x = concatenate({treeExps[sp[sp_size-2]].second, treeExps[sp[lcn]].first, treeExps[sp[1]].second});
    }else{
      x = concatenate({treeExps[sp[1]].second, treeExps[sp[lcn]].first, treeExps[sp[sp_size - 2]].second});
    }
    return x;
  }else{
    vector<cnn::real> x_values(params_.h1dimension() * 3, 0.); 
    Expression x = input(cg, {params_.h1dimension() * 3}, &x_values);
    cg.incremental_forward();
    return x;
  }
}

Expression RelLSTMModel::calc_sp_expression(cnn::ComputationGraph& cg, TableCell* cell, vector<Expression> &seqExps, bool reverse){
  assert(cell->row() > cell->col());
  Table* table = &(cell->table());
  deque<int> sp;
  if(reverse){
    sp = table->get_path(cell->row(), cell->col());
  }else{
    sp = table->get_path(cell->col(), cell->row());
  }
  int sp_size = sp.size();
  Expression depFwdMax, depRevMax;
  if(sp_size > 0){
    assert(sp_size > 2);
    sp.push_front(NEGATIVE_DEPENDENCY_ID);
    sp.push_back(NEGATIVE_DEPENDENCY_ID);
    sp_size += 2;
    // find least common node
    int lcn = -1;
    for(int i = 1;i < sp_size;i += 2){
      if(i == 1){
        if(sp[i-1] % 2 == sp[i+1] % 2){
          lcn = i;
        }
      }else{
        if(sp[i-1] % 2 != sp[i+1] % 2){
          assert(lcn == -1);
          lcn = i;
        }
      }
    }
    sp[sp.size()-1] += 1; // to mimick rev neg
    assert(lcn > 0 && lcn < sp_size - 1);
    //cerr << "lcn: " <<  dict_.get_repr_string(table->word(sp[lcn])) << "/" << dict_.get_pos_string(table->pos(sp[lcn])) << endl;
    spDepLstm_.new_graph(cg);
    spDepLstm_.start_new_sequence();
    for(int i = 1;i <= lcn;i += 2){
      // Expression depFwd = spDepLstm_.add_input(get_expression(cg, table, sp[i], seqExps));
      Expression depFwd;
      if(params_.dep_dimension() > 0){
        if(i == lcn){
          depFwd = spDepLstm_.add_input(concatenate({get_expression(cg, table, sp[i], seqExps), dropout_lookup(cg, d2i, NEGATIVE_DEPENDENCY_ID)}));
        }else{
          depFwd = spDepLstm_.add_input(concatenate({get_expression(cg, table, sp[i], seqExps), dropout_lookup(cg, d2i, sp[i+1])}));
        }
      }else{
        depFwd = spDepLstm_.add_input(get_expression(cg, table, sp[i], seqExps));
      }
      if(i == 1){
        depFwdMax = depFwd;
      }else{
        depFwdMax = max(depFwd, depFwdMax);
      }
    }
    ispDepLstm_.new_graph(cg);
    ispDepLstm_.start_new_sequence();
    for(int i = sp_size - 2;i >= lcn;i -= 2){
      // Expression depRev = ispDepLstm_.add_input(get_expression(cg, table, sp[i], seqExps));
      Expression depRev;
      if(params_.dep_dimension() > 0){
        if(i == lcn){
          depRev = ispDepLstm_.add_input(concatenate({get_expression(cg, table, sp[i], seqExps), dropout_lookup(cg, d2i, NEGATIVE_DEPENDENCY_ID)}));
        }else{
          depRev = ispDepLstm_.add_input(concatenate({get_expression(cg, table, sp[i], seqExps), dropout_lookup(cg, d2i, sp[i-1]-1)}));
        }
      }else{
        depRev = ispDepLstm_.add_input(get_expression(cg, table, sp[i], seqExps));
      }
      if(i == sp_size - 2){
        depRevMax = depRev;
      }else{
        depRevMax = max(depRev, depRevMax);
      }
    }
    // ispDepLstm_.new_graph(cg);
    // ispDepLstm_.start_new_sequence();
    // for(int i = lcn;i >= 1;i -= 2){
    //   Expression depFwd;
    //   if(i == 1){
    //     depFwd = ispDepLstm_.add_input(concatenate({get_expression(cg, table, sp[i], seqExps), dropout_lookup(cg, d2i, NEGATIVE_DEPENDENCY_ID)}));
    //   }else{
    //     depFwd = ispDepLstm_.add_input(concatenate({get_expression(cg, table, sp[i], seqExps), dropout_lookup(cg, d2i, sp[i-1])}));
    //   }
    //   depFwdMax = max(depFwd, depFwdMax);
    // }
    // ispDepLstm_.new_graph(cg);
    // ispDepLstm_.start_new_sequence();
    // for(int i = lcn;i <= sp_size - 2;i += 2){
    //   Expression depRev;
    //   if(i == sp_size - 2){
    //     depRev = ispDepLstm_.add_input(concatenate({get_expression(cg, table, sp[i], seqExps), dropout_lookup(cg, d2i, NEGATIVE_DEPENDENCY_ID)}));
    //   }else{
    //     depRev = ispDepLstm_.add_input(concatenate({get_expression(cg, table, sp[i], seqExps), dropout_lookup(cg, d2i, sp[i+1]-1)}));
    //   }
    //   depRevMax = max(depRev, depRevMax);
    // }
    Expression relExp = concatenate({depFwdMax, depRevMax});
    // vector<float> v = as_vector(relExp.value());
    // for(float f:v){
    //   assert(!isnan(f));
    // }
    return relExp;
  }else{
    vector<cnn::real> x_values(params_.h1dimension() * 2, 0.); 
    Expression x = input(cg, {params_.h1dimension() * 2}, &x_values);
    cg.incremental_forward();
    return x;
  }
}

void RelLSTMModel::calc_seq_expressions(cnn::ComputationGraph &cg, Table* table, vector<pair<Expression, Expression> >& seqExps){
  eseqLstm_.new_graph(cg);
  eseqLstm_.start_new_sequence();
  int ent_num = table->size();
  for(int i = 0; i < ent_num;++i){
    seqExps[i].first = eseqLstm_.add_input(get_expression(cg, table, i));
  }
  // vector<Expression> init;
  // init.insert(init.end(), eseqLstm_.c.back().begin(), eseqLstm_.c.back().end());
  // init.insert(init.end(), eseqLstm_.h.back().begin(), eseqLstm_.h.back().end());
  ieseqLstm_.new_graph(cg);
  // ieseqLstm_.start_new_sequence(init);
  ieseqLstm_.start_new_sequence();
  for(int i = ent_num - 1;i >= 0;--i){
    seqExps[i].second = ieseqLstm_.add_input(get_expression(cg, table, i));
  }
}

int RelLSTMModel::calc_butree_expressions(const TreeNode* node, int& node_idx, unordered_map<int, int>& sent2lstm, unordered_map<int, Expression>& expressions, cnn::ComputationGraph &cg, Table* table, vector<Expression>& seqExps, unordered_set<int> &sp_nodes){
  const Tree& tree = table->sentence().dep_tree();
  vector<pair<int, int> > children;
  bool in_sp = (sp_nodes.find(node->id()) != sp_nodes.end());
  for(int child:node->children()){
    int child_id = calc_butree_expressions(tree.node(child), node_idx, sent2lstm, expressions, cg, table, seqExps, sp_nodes);
    if(in_sp && sp_nodes.find(tree.node(child)->id()) != sp_nodes.end()){
       children.push_back(make_pair(1, child_id));
    }else{      
      children.push_back(make_pair(0, child_id));
    }
    //children.push_back(make_pair(tree.node(child)->dep()/2, child_id));
  }
  Expression input;
  if(params_.dep_dimension() > 0){
    input = concatenate({get_expression(cg, table, node->id(), seqExps), dropout_lookup(cg, d2i, node->dep())});
  }else{
    input = get_expression(cg, table, node->id(), seqExps);
  }
  Expression expression = subTreeLstm_.add_input(children, input);
  expressions.insert(make_pair(node_idx, expression));
  sent2lstm.insert(make_pair(node->id(), node_idx));
  node_idx++;
  return node_idx - 1;
}


void RelLSTMModel::calc_tdtree_expressions(const TreeNode* node, const TreeNode *parent_node, int parent, int& node_idx, unordered_map<int, int>& sent2lstm, unordered_map<int, Expression>& expressions, cnn::ComputationGraph &cg, Table* table, vector<Expression>& seqExps, unordered_set<int>& sp_nodes){
  const Tree& tree = table->sentence().dep_tree();
  vector<pair<int, int> > children;
  bool in_sp = (sp_nodes.find(node->id()) != sp_nodes.end());
  if(parent >= 0){
    if(in_sp && sp_nodes.find(parent_node->id()) != sp_nodes.end()){
      children.push_back(make_pair(1, parent));
    }else{
      children.push_back(make_pair(0, parent));
    }
    //children.push_back(make_pair(node->dep()/2, parent));
  }else if(node_idx > 0){
    // should not happen
    if(in_sp && sp_nodes.find(node_idx - 1) != sp_nodes.end()){
       children.push_back(make_pair(1, node_idx - 1));
    }else{
      children.push_back(make_pair(0, node_idx - 1));
    }
    // children.push_back(make_pair(0, node_idx - 1));
    //children.push_back(make_pair(2, node_idx - 1));     
  }
  Expression input;
  if(params_.dep_dimension() > 0){
    input = concatenate({get_expression(cg, table, node->id(), seqExps), dropout_lookup(cg, d2i, node->dep())});
  }else{
    input = get_expression(cg, table, node->id(), seqExps);
  }
  Expression expression = isubTreeLstm_.add_input(children, input);
  expressions.insert(make_pair(node_idx, expression));
  sent2lstm.insert(make_pair(node->id(), node_idx));
  int cparent = node_idx;
  node_idx++;
  for(int child:node->children()){
    calc_tdtree_expressions(tree.node(child), node, cparent, node_idx, sent2lstm, expressions, cg, table, seqExps, sp_nodes);
  }
  return;
}

void RelLSTMModel::calc_subtree_expressions(cnn::ComputationGraph &cg, TableCell* cell, vector<Expression>& seqExps, vector<pair<Expression, Expression> >& treeExps){
  Table* table = &(cell->table());
  const Tree& tree = table->sentence().dep_tree();
  
  deque<int> sp = table->get_path(cell->row(), cell->col());
  int sp_size = sp.size();
  if(sp_size == 0)return;
  sp.push_front(NEGATIVE_DEPENDENCY_ID);
  sp.push_back(NEGATIVE_DEPENDENCY_ID);
  sp_size += 2;
  // find least common node
  unordered_set<int> sp_nodes;
  int lcn = -1;
  for(int i = 1;i < sp_size;i += 2){
    sp_nodes.insert(sp[i]);
    if(i == 1){
      if(sp[i-1] % 2 == sp[i+1] % 2){
          lcn = i;
      }
    }else{
      if(sp[i-1] % 2 != sp[i+1] % 2){
        assert(lcn == -1);
        lcn = i;
      }
    }
  }
  assert(lcn != -1);
  unordered_map<int, int> sent2bulstm;
  unordered_map<int, Expression> buExpressions;  
  subTreeLstm_.new_graph(cg);
  subTreeLstm_.start_new_sequence();
  int node_idx = 0;
  assert(tree.root() != nullptr);
  if(params_.use_sp_sub_tree_exp()){
    calc_butree_expressions(tree.node(sp[lcn]), node_idx, sent2bulstm, buExpressions, cg, table, seqExps, sp_nodes);
  }else{
    calc_butree_expressions(tree.root(), node_idx, sent2bulstm, buExpressions, cg, table, seqExps, sp_nodes);
  }
  unordered_map<int, int> sent2tdlstm;
  unordered_map<int, Expression> tdExpressions;  
  isubTreeLstm_.new_graph(cg);
  isubTreeLstm_.start_new_sequence();
  node_idx = 0;
  if(params_.use_sp_sub_tree_exp()){
    calc_tdtree_expressions(tree.node(sp[lcn]), nullptr, -1, node_idx, sent2tdlstm, tdExpressions, cg, table, seqExps, sp_nodes);
  }else{
    calc_tdtree_expressions(tree.root(), nullptr, -1, node_idx, sent2tdlstm, tdExpressions, cg, table, seqExps, sp_nodes);
  }
  int ent_num = table->size();
  for(int i = 0; i < ent_num;++i){
    if(sp_nodes.find(i) != sp_nodes.end()){
      assert(sent2bulstm.find(i) != sent2bulstm.end());
      assert(sent2tdlstm.find(i) != sent2tdlstm.end());
    }
    if(sent2bulstm.find(i) != sent2bulstm.end() && sent2tdlstm.find(i) != sent2tdlstm.end()){
      treeExps[i] = make_pair(buExpressions[sent2bulstm[i]], tdExpressions[sent2tdlstm[i]]);
    }else{
      treeExps[i] = make_pair(Expression(), Expression());
    }
  }
}


int RelLSTMModel::calc_butree_expressions(const TreeNode* node, int& node_idx, unordered_map<int, int>& sent2lstm, unordered_map<int, Expression>& expressions, cnn::ComputationGraph &cg, Table* table){
  const Tree& tree = table->sentence().dep_tree();
  vector<pair<int, int> > children;
  for(int child:node->children()){
    int child_id = calc_butree_expressions(tree.node(child), node_idx, sent2lstm, expressions, cg, table);
    children.push_back(make_pair(0, child_id));
    //children.push_back(make_pair(tree.node(child)->dep()/2, child_id));
  }
  Expression input;
  if(params_.dep_dimension() > 0){
    input = concatenate({get_expression(cg, table, node->id()), dropout_lookup(cg, d2i, node->dep())});
  }else{
    input = get_expression(cg, table, node->id());
  }
  Expression expression = fullTreeLstm_.add_input(children, input);
  expressions.insert(make_pair(node_idx, expression));
  sent2lstm.insert(make_pair(node->id(), node_idx));
  node_idx++;
  return node_idx - 1;
}


void RelLSTMModel::calc_tdtree_expressions(const TreeNode* node, const TreeNode *parent_node, int parent, int& node_idx, unordered_map<int, int>& sent2lstm, unordered_map<int, Expression>& expressions, cnn::ComputationGraph &cg, Table* table){
  const Tree& tree = table->sentence().dep_tree();
  vector<pair<int, int> > children;
  if(parent >= 0){
    children.push_back(make_pair(0, parent));
    // children.push_back(make_pair(node->dep()/2, parent));
  }else if(node_idx > 0){
    children.push_back(make_pair(2, node_idx - 1));
  }
  Expression input;
  if(params_.dep_dimension() > 0){
    input = concatenate({get_expression(cg, table, node->id()), dropout_lookup(cg, d2i, node->dep())});
  }else{
    input = get_expression(cg, table, node->id());
  }
  Expression expression = ifullTreeLstm_.add_input(children, input);
  expressions.insert(make_pair(node_idx, expression));
  sent2lstm.insert(make_pair(node->id(), node_idx));
  int cparent = node_idx;
  node_idx++;
  for(int child:node->children()){
    calc_tdtree_expressions(tree.node(child), node, cparent, node_idx, sent2lstm, expressions, cg, table);
  }
  return;
}

void RelLSTMModel::calc_fulltree_expressions(cnn::ComputationGraph &cg, Table* table, vector<pair<Expression, Expression> >& treeExps){
  const Tree& tree = table->sentence().dep_tree();
  unordered_map<int, int> sent2bulstm;
  unordered_map<int, Expression> buExpressions;
  fullTreeLstm_.new_graph(cg);
  fullTreeLstm_.start_new_sequence();
  int node_idx = 0;
  calc_butree_expressions(tree.root(), node_idx, sent2bulstm, buExpressions, cg, table);
  unordered_map<int, int> sent2tdlstm;
  unordered_map<int, Expression> tdExpressions;  
  ifullTreeLstm_.new_graph(cg);
  ifullTreeLstm_.start_new_sequence();
  node_idx = 0;
  calc_tdtree_expressions(tree.root(), nullptr, -1, node_idx, sent2tdlstm, tdExpressions, cg, table);
  int ent_num = table->size();
  for(int i = 0; i < ent_num;++i){
    assert(sent2bulstm.find(i) != sent2bulstm.end());
    assert(sent2tdlstm.find(i) != sent2tdlstm.end());
    treeExps[i] = make_pair(buExpressions[sent2bulstm[i]], tdExpressions[sent2tdlstm[i]]);
  }
}

} /* namespace coin */
