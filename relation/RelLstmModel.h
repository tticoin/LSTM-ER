/*
 * RelLstmModel.h
 *
 *  Created on: 2015/09/13
 *      Author: miwa
 */

#ifndef RELLSTMMODEL_H_
#define RELLSTMMODEL_H_

#include "cnn/zlstm.h"
#include "cnn/treelstm.h"
#include "cnn/deptreelstm.h"
#include "cnn/training.h"
#include "Parameter.h"
#include "Document.h"
#include "Table.h"

namespace coin {

class RelLSTMModel {
private:
  enum{S2H=0, SBH, H2E, BE, P2H, B2H, D2H, T2H, F2H, BH, H2R, BR};
  const Parameters& params_;
  Dictionary& dict_;
  bool do_update_;
  int updates_;
  int iterations_;
  int word_dim_;
  int pos_dim_;
  int seq_dim_;
  int pair_dim_;
  int rel_dim_;
  cnn::Model model_;
  cnn::Trainer *sgd_;
  cnn::ZLSTMBuilder eseqLstm_, ieseqLstm_;
  cnn::DepTreeLSTMBuilder fullTreeLstm_, ifullTreeLstm_;
  cnn::ZLSTMBuilder depLstm_, idepLstm_;
  cnn::ZLSTMBuilder spDepLstm_, ispDepLstm_;
  cnn::DepTreeLSTMBuilder subTreeLstm_, isubTreeLstm_;
  cnn::DepTreeLSTMBuilder spTreeLstm_, ispTreeLstm_;
  cnn::LookupParameters *w2i, *p2i, *d2i, *e2i, *wn2i; //word, pos, dependency, label, wordnet
  cnn::LookupParameters *e2s, *r2s; // reference distributions
  cnn::Parameters *s2h, *s2h_bias, *h2e, *h2e_bias;
  cnn::Parameters *p2h, *b2h, *d2h, *t2h, *f2h, *h_bias;
  cnn::Parameters *h2r, *h2r_bias;
  vector<Expression> param_vars_;
  void read_w2v();
  void read_p2v();
  void init_entity_scores();
  void init_relation_scores();
  void init_params(cnn::ComputationGraph &cg);
  Expression dropout_const_lookup(cnn::ComputationGraph& cg, cnn::LookupParameters *param, int i){
    return dropout_output(const_lookup(cg, param, i), params_.idropout());
  }
  Expression dropout_lookup(cnn::ComputationGraph& cg, cnn::LookupParameters *param, int i){
    return dropout_output(lookup(cg, param, i), params_.idropout());
  }
  Expression dropout_output(Expression exp, double p){
    if(do_update_){
      return dropout(exp, p);
    }else{
      return exp;
    }
  }
  Expression get_expression(cnn::ComputationGraph& cg, Table *table, int i);
  Expression get_expression(cnn::ComputationGraph& cg, Table *table, int i, vector<Expression> &seqExps);
  Expression calc_region_expression(cnn::ComputationGraph& cg, Table* table, vector<Expression>& seqExps, int start, int end);
  Expression calc_pair_expression(cnn::ComputationGraph& cg, TableCell* cell, vector<Expression>& seqExps, bool reverse = false);
  void calc_seq_expressions(cnn::ComputationGraph& cg, Table *table, vector<pair<Expression, Expression> >& seqExps);
  
  void calc_fulltree_expressions(cnn::ComputationGraph& cg, Table *table, vector<pair<Expression, Expression> >& treeExps);
  void calc_subtree_expressions(cnn::ComputationGraph& cg, TableCell *cell, vector<Expression>& seqExps, vector<pair<Expression, Expression> >& treeExps);
  Expression calc_rel_expression(cnn::ComputationGraph& cg, TableCell* cell, vector<Expression>& seqExps, bool reverse = false);
  Expression calc_sp_tree_expression(cnn::ComputationGraph& cg, TableCell* cell, vector<Expression> &seqExps, bool revese = false);
  Expression calc_sp_subtree_expression(cnn::ComputationGraph& cg, TableCell* cell, vector<pair<Expression, Expression> >& treeExps, bool reverse = false);
  Expression calc_sp_expression(cnn::ComputationGraph& cg, TableCell* cell, vector<Expression> &seqExps, bool reverse = false);

  int calc_butree_expressions(const TreeNode* node, int& node_idx, unordered_map<int, int>& sent2lstm, unordered_map<int, Expression>& expressions, cnn::ComputationGraph &cg, Table* table, vector<Expression>& seqExps, unordered_set<int> &sp_nodes);
  void calc_tdtree_expressions(const TreeNode* node, const TreeNode *parent_node, int parent, int& node_idx, unordered_map<int, int>& sent2lstm, unordered_map<int, Expression>& expressions, cnn::ComputationGraph &cg, Table* table, vector<Expression>& seqExps, unordered_set<int> &sp_nodes);

  
  int calc_butree_expressions(const TreeNode* node, int& node_idx, unordered_map<int, int>& sent2lstm, unordered_map<int, Expression>& expressions, cnn::ComputationGraph &cg, Table* table);
  void calc_tdtree_expressions(const TreeNode* node, const TreeNode *parent_node, int parent, int& node_idx, unordered_map<int, int>& sent2lstm, unordered_map<int, Expression>& expressions, cnn::ComputationGraph &cg, Table* table);
  void predict(vector<Table*> &tables, bool do_update, bool output, bool ent_only);
public:
  // Default constructor
  RelLSTMModel(const Parameters& params, Dictionary& dict);
  // Constructor for model loading
  RelLSTMModel(const Parameters& params, Dictionary& dict, std::ifstream &is);
  void save_model(std::ofstream &os);
  // training 
  void update(vector<Table*> &tables, bool ent_only=false){
    predict(tables, true, false, ent_only);
  }
  // prediction 
  void predict(vector<Table*> &tables, bool ent_only=false){
    predict(tables, false, false, ent_only);
  }
  // prediction with output
  void output(vector<Table*> &tables, bool ent_only=false){
    predict(tables, false, true, ent_only);
  }
  virtual ~RelLSTMModel(){
    delete sgd_;
  };
};

} /* namespace coin */

#endif /* RELLSTMMODEL_H_ */
