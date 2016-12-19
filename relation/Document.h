/*
 * Document.h
 *
 *  Created on: 2014/07/05
 *      Author: miwa
 */

#ifndef _DOCUMENT_H_
#define _DOCUMENT_H_

#include <list>
#include <vector>
#include <deque>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <unicode/unistr.h>
#include <cassert>
#include <algorithm>
#include <iostream>

#include "Parameter.h"
using std::list;
using std::vector;
using std::deque;
using std::string;
using std::unordered_set;
using std::unordered_map;

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/unordered_set.hpp>
#include <boost/serialization/unordered_map.hpp>

namespace coin {

class Node {
protected:
	int start_;
	int end_;
public:
	Node(int start, int end):start_(start), end_(end){}
	int start() const {
		return start_;
	}
	int end() const {
		return end_;
	}
	bool inside_of(const Node& n2) const{
		return n2.start() <= start() && end() <= n2.end();
	}
	bool overlap(const Node& n2) const{
		if(start() <= n2.start() && n2.start() < end())return true;
		if(n2.start() <= start() && start() < n2.end())return true;
		return false;
	}
};


class Constituent : public Node{
protected:
	string id_;
	Constituent *head_;
	unordered_map<string, string> attrs_;
	const ParseParameters& parse_;
public:
	Constituent(int start, int end, unordered_map<string, string>& attrs, const ParseParameters& parse):
    Node(start, end), id_(attrs[parse.id_attr()]), head_(nullptr), attrs_(attrs), parse_(parse) {
	}
	const unordered_map<string, string>& attrs() const {
		return attrs_;
	}
	const string& id() const {
		return id_;
	}
	void set_head(Constituent *head){
	  head_ = head;
	}
	Constituent *head(){
	  return head_;
	}
};

class Term;
class Dictionary;
class Sentence;

class Word : public Constituent{
private:
	string base_;
  string surface_;
  string repr_;
	string pos_;
  string wn_;
	string label_;
	vector<Term*> terms_;
	const Term* direct_term_;
	unordered_map<string, unordered_set<Word*> > dependencies_;
	int repr_id_;
	int pos_id_;
  int wn_id_;
	int label_id_;
	unordered_map<int, unordered_set<Word*> > dependencies_ids_;
public:
	Word(const Parameters& params, int start, int end, string surface, unordered_map<string, string>& attrs, const ParseParameters& parse):
		Constituent(start, end, attrs, parse), base_(attrs[parse.base_attr()]), surface_(surface), 
		pos_(attrs[parse.pos_attr()]), wn_(""), label_(""), direct_term_(nullptr), repr_id_(-1), pos_id_(-1), wn_id_(-1), label_id_(-1){
    if(attrs.find(parse.wn_attr()) != attrs.end()){
      wn_ = attrs[parse.wn_attr()];
    }
    if(params.use_base()){
      repr_ = base_;
    }else{
      repr_ = surface_;
      if(params.use_lowercase()){
        transform(repr_.begin(), repr_.end(), repr_.begin(), ::tolower);
      }
    }
  }
	const string& repr() const {
		return repr_;
	}
	const string& pos() const {
		return pos_;
	}
	const string& wn() const {
		return wn_;
	}
	const string& label() const {
		return label_;
	}
	void add(Term* term){
		terms_.push_back(term);
	}
	void set_label(const string& label) {
		label_ = label;
	}
	vector<Term*>& terms(){
		return terms_;
	}
	void set_dependency(string& type, Word *word){
    if(word == this)return;
    if(dependencies_.find(type) == dependencies_.end()){
      dependencies_[type] = unordered_set<Word*>();
    }
	  dependencies_[type].insert(word);
	}
	void set_direct_term(const Term* direct_term) {
		direct_term_ = direct_term;
	}
	const Term* direct_term() const {
		return direct_term_;
	}
	const int repr_id() const{
	  return repr_id_;
	}
	const int pos_id() const{
	  return pos_id_;
	}
	const int wn_id() const{
	  return wn_id_;
	}
	const int label_id() const{
	  return label_id_;
	}
  unordered_map<string, unordered_set<Word*> >& dependencies(){
    return dependencies_;
  }
	unordered_map<int, unordered_set<Word*> >& dependencies_ids(){
	  return dependencies_ids_;
	}
	void update(const Parameters& params, Dictionary* dict, unordered_map<string, unsigned>& word_counts, unordered_map<string, unsigned>& dep_counts);
	void apply(const Dictionary* dict, Sentence *sentence);
};

class Term : public Node{
private:
	string id_;
	string type_;
	vector<Word*> words_;
public:
	Term(int start, int end, string id, string type):
		Node(start, end), id_(id), type_(type){}
	Term(const Term& term):
		Node(term.start(), term.end()), id_(term.id()), type_(term.type()){}
	const string& id() const {
		return id_;
	}
	const string& type() const {
		return type_;
	}
	void add(Word* word){
		words_.push_back(word);
	}
	void remove(Word* word){
		words_.erase(std::remove(words_.begin(), words_.end(), word), words_.end());
	}
	const vector<Word*>& words() const {
		return words_;
	}
	bool is_edge(Word* word){
    if(words_.size() <= 0){
      std::cerr << "no words for term " << id_ << endl;
    }
		assert(words_.size() > 0);
		if(words_[0] == word || words_[words_.size() - 1] == word){
			return true;
		}
		return false;
	}
};


class Relation{
private:
	string id_;
	string type_;
	string arg1_;
	string arg2_;
public:
	Relation(string id, string type, string arg1, string arg2):
		id_(id), type_(type), arg1_(arg1), arg2_(arg2){
    if(arg1 == arg2){
      std::cerr << "no words for term " << id_ << endl;
    }
    assert(arg1 != arg2);
  }
	Relation(const Relation& relation):
		id_(relation.id()), type_(relation.type()), arg1_(relation.arg1()), arg2_(relation.arg2()){}
	const string& arg1() const {
		return arg1_;
	}
	const string& arg2() const {
		return arg2_;
	}
	const string& id() const {
		return id_;
	}
	const string& type() const {
		return type_;
	}
};

class IndexRelation{
private:
  string id_;
	int arg1_;
	int arg2_;
	int type_;
public:
	IndexRelation(const string &id, int arg1, int arg2, const string& type, Dictionary *dict);
  IndexRelation(const string &id, int arg1, int arg2, const string& type, const Dictionary *dict);
  string id() const{
    return id_;
  }
	int arg1() const {
		return arg1_;
	}
	int arg2() const {
		return arg2_;
	}
	const int& type() const {
		return type_;
	}
};

class Document;

class TreeNode {
private:
  int parent_ = -1;
  int dep_ = NEGATIVE_DEPENDENCY_ID;
  int id_;
  vector<int> children_;
public:
  TreeNode(int id):id_(id){}
  int id() const{
    return id_;
  }
  void set_parent(int parent){
    parent_ = parent;
  }
  int parent() const{
    return parent_;
  }
  void set_dep(int dep){
    dep_ = dep;
  }
  int dep() const{
    return dep_;
  }
  void add_child(int child){
    children_.push_back(child);
  }
  const vector<int>& children() const{
    return children_;
  }
};

class Tree{
private:
  TreeNode *root_ = nullptr;
  vector<TreeNode*> nodes_;
public:
  Tree(){}
  void add_node(TreeNode* node){
    nodes_.push_back(node);
  }
  const TreeNode *node(int idx) const{
    return nodes_[idx];
  }
  TreeNode *node(int idx){
    return nodes_[idx];
  }
  void set_root(TreeNode *root){
    if(root_ != nullptr){
      std::cerr << "root is already set" << endl;
    }
    assert(root_ == nullptr);
    root_ = root;
  }
  const TreeNode *root() const{
    return root_;
  }
  ~Tree(){
    for(TreeNode* node:nodes_){
      delete node;
    }
  }
};  

class Sentence: public Node{
private:
  const Document &doc_;
	string id_;
  int missing_terms_;
  int missing_rels_;
	vector<Word*> words_;
	vector<Term*> terms_;
	vector<Relation*> rels_;
  unordered_map<string,int> term_rel_counts_;
	vector<IndexRelation> index_rels_;
	vector<vector<std::pair<int, int> > > paths_;
	unordered_set<string> term_ids_;
  Tree dep_tree_;
	unordered_map<string, unordered_map<string, Constituent*>> nodes_;
public:
	Sentence(int start, int end, string id, const Document& doc):
		Node(start, end), doc_(doc), id_(id), missing_terms_(0), missing_rels_(0){}
	virtual ~Sentence();
	void add(Word* word){
		words_.push_back(word);
	}
	void add(Constituent* cons, string type){
		if(nodes_.find(type) == nodes_.end()){
			nodes_[type] = unordered_map<string, Constituent*>();
		}
		nodes_[type][cons->id()] = cons;
	}
	void add(Term* term){
    if(contains_term(term->id())){
      std::cerr << "duplicated term " << term->id() << endl;
    }    
		assert(!contains_term(term->id()));
		terms_.push_back(term);
		term_ids_.insert(term->id());
	}
	void add(Relation* rel){
		rels_.push_back(rel);
    if(term_rel_counts_.find(rel->arg1()) == term_rel_counts_.end()){
      term_rel_counts_[rel->arg1()] = 0;
    }
    if(term_rel_counts_.find(rel->arg2()) == term_rel_counts_.end()){
      term_rel_counts_[rel->arg2()] = 0;
    }
    term_rel_counts_[rel->arg1()]++;
    term_rel_counts_[rel->arg2()]++;
	}
  void inc_missing_terms(){
    missing_terms_++;
  }
  void inc_missing_rels(){
    missing_rels_++;
  }
  int missing_terms() const{
    return missing_terms_;
  }
  int missing_rels() const{
    return missing_rels_;
  }
	bool contains_term(const string& term_id){
		return term_ids_.find(term_id) != term_ids_.end();
	}
	const vector<Word*>& words() const {
		return words_;
	}
	void build_word_annotations();
	void update_relations(Dictionary *dict);
	void apply_relations(const Dictionary *dict);
	void build_dependencies(const ParseParameters& parse);
	vector<Word*>& words(){
	  return words_;
	}
	const vector<Relation*>& rels() const {
		return rels_;
	}
	const vector<Term*>& terms() const {
		return terms_;
	}
	const vector<IndexRelation>& index_rels() const {
		return index_rels_;
	}
  const Document& doc() const{
    return doc_;
  }
  const Tree& dep_tree() const{
    return dep_tree_;
  }
	void calculate_shortest_paths();
  void calculate_dep_tree(const Dictionary* dict);
	deque<int> get_path(int from, int to) const;
};

class Table;

class Document {
private:
  const Parameters& params_;
  bool has_annotation_;
  string id_;
	UnicodeString *text_;
	UnicodeString* read_file(const string& file) const;
	void read_text(const string& file){
		//TODO: binary_mode
		text_ = read_file(file);
	}
	void read_annotation(const string& file);
	void read_parse(const string& file, const ParseParameters& parse);
	string convert(const UnicodeString& str) const;
	unordered_map<string, Term*> terms_;
	unordered_map<string, Relation*> relations_;
	vector<Sentence*> sentences_;
	vector<Table*> tables_;
public:
	Document(const Parameters& params, const string& base);
	virtual ~Document();
	const Term& term(string term_id) const{
		return *terms_.at(term_id);
	}
  string text(int start, int len) const{
    if(text_->length() < start + len){
      std::cerr << "index is larger than text size." << std::endl;
    }    
    assert(text_->length() >= start + len);
		return convert(text_->tempSubString(start, len));
	}
	vector<Sentence*>& sentences(){
	  return sentences_;
	}
	vector<Table*>& tables();
  const string& id() const{
    return id_;
  }
};

class DocumentCollection{
private:
	list<Document*> documents_;
	void add_document(Document* doc);
public:
	DocumentCollection(const Parameters& params, const string& dir);
	list<Document*> documents(){
	  return documents_;
	}
	virtual ~DocumentCollection();
	vector<Table*> collect_tables();
};

class DictEntry{
private:
  int types_;
  unordered_map<string, int> dict_;
  vector<string> rev_dict_;
public:
  DictEntry():types_(0){}
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & types_;
    ar & dict_;
    ar & rev_dict_;
  }
  int get_id(const string& target){
    if(dict_.find(target) == dict_.end()){
      dict_.insert(make_pair(target, types_++));
      rev_dict_.push_back(target);
    }
    return dict_[target];
  }
  int get_id(const string& target) const{
    if(dict_.find(target) == dict_.end()){
      return -1;
    }
    return dict_.at(target);
  }
  string get_string(const int index) const{
    return rev_dict_.at(index);
  }
  int types() const{
    return types_;
  }
  const unordered_map<string, int>& dict(){
    return dict_;
  }
};


class Dictionary{
private:
  DictEntry repr_entry_;
  DictEntry pos_entry_;
  DictEntry wn_entry_;
  DictEntry dep_entry_;
  DictEntry ent_entry_;
  DictEntry rel_entry_;
  unordered_set<int> entity_labels_;
  unordered_map<int, int> begin_labels_;
  unordered_map<int, unordered_set<int> > prev_entities_;
  unordered_map<int, unordered_set<int> > next_entities_;
  unordered_map<int, unordered_set<int> > prev_first_entities_;
  unordered_map<int, unordered_set<int> > next_last_entities_;
public:
  Dictionary(){
    int neg_dep_id = dep_entry_.get_id("");
    int neg_ent_id = ent_entry_.get_id("O");
    int neg_rel_id = rel_entry_.get_id(NEGATIVE_RELATION);
    int neg_wn_id = wn_entry_.get_id("0");
    dep_entry_.get_id("UNK");
    dep_entry_.get_id(REVERSE_DEP_HEADER+"UNK");
    assert(neg_dep_id == NEGATIVE_DEPENDENCY_ID);
    assert(neg_ent_id == NEGATIVE_ENTITY_ID);
    assert(neg_rel_id == NEGATIVE_RELATION_ID);
    assert(neg_wn_id == NEGATIVE_WORDNET_ID);
  }
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & repr_entry_;
    ar & pos_entry_;
    ar & wn_entry_;
    ar & dep_entry_;
    ar & ent_entry_;
    ar & rel_entry_;
    ar & entity_labels_;
    ar & begin_labels_;
    ar & prev_entities_;
    ar & next_entities_;
    ar & prev_first_entities_;
    ar & next_last_entities_;
  }
  void update(const Parameters& params, DocumentCollection& collection);
  void apply(DocumentCollection& collection) const;
  void apply(Document& doc) const;
  int get_repr_id(const string& repr){
    return repr_entry_.get_id(repr);
  }
  int get_pos_id(const string& pos){
    return pos_entry_.get_id(pos);
  }
  int get_wn_id(const string& wn){
    return wn_entry_.get_id(wn);
  }
  int get_dep_id(const string& dep){
    return dep_entry_.get_id(dep);
  }
  int get_ent_id(const string& ent){
    if(ent == ""){
      cerr << "id is empty" << endl;
    }
    assert(ent != "");
    return ent_entry_.get_id(ent);
  }
  int get_rel_id(const string& rel){
    return rel_entry_.get_id(rel);
  }
  string get_repr_string(const int repr) const {
    return repr_entry_.get_string(repr);
  }
  string get_pos_string(const int pos) const {
    return pos_entry_.get_string(pos);
  }
  string get_wn_string(const int wn) const {
    return wn_entry_.get_string(wn);
  }
  string get_dep_string(const int dep) const {
    return dep_entry_.get_string(dep);
  }
  string get_ent_string(const int ent) const {
    return ent_entry_.get_string(ent);
  }
  string get_rel_string(const int rel) const {
    return rel_entry_.get_string(rel);
  }
  int repr_types() const{
    return repr_entry_.types();
  }
  int pos_types() const{
    return pos_entry_.types();
  }
  int wn_types() const{
    return wn_entry_.types();
  }
  int dep_types() const{
    return dep_entry_.types();
  }
  int ent_types() const{
    return ent_entry_.types();
  }
  int rel_types() const{
    return rel_entry_.types();
  }
  int get_repr_id(const string& repr) const{
    return repr_entry_.get_id(repr);
  }
  int get_pos_id(const string& pos) const{
    return pos_entry_.get_id(pos);
  }
  int get_wn_id(const string& wn) const{
    return wn_entry_.get_id(wn);
  }
  int get_dep_id(const string& dep) const{
    return dep_entry_.get_id(dep);
  }
  int get_ent_id(const string& ent) const{
    return ent_entry_.get_id(ent);
  }
  int get_rel_id(const string& rel) const{
    return rel_entry_.get_id(rel);
  }
  bool is_entity_label(int id) const{
    return entity_labels_.find(id) != entity_labels_.end();
  }
  bool is_negative_relation(int id) const{
    return id == NEGATIVE_RELATION_ID;
  }
  bool is_reverse_relation(int id) const{
    if(is_negative_relation(id)){
      return false;
    }else{
      return id % 2 == 0;
    }
  }
  int reverse_relation(int id) const{
    if(is_negative_relation(id)){
      return id;
    }else if(id % 2 == 1){
      return id+1;
    }else{
      return id-1;
    }
  }
  const int get_begin_label(int label_id) const{        
    if(get_ent_string(label_id)[0] != 'L' && get_ent_string(label_id)[0] != 'U'){
      cerr << "Not beginning label" << endl;
    }
    assert(get_ent_string(label_id)[0] == 'L' || get_ent_string(label_id)[0] == 'U');
    return begin_labels_.at(label_id);
  }
  const unordered_set<int>& prev_entities(int id, bool first) const{
    if(first){
      return prev_first_entities_.at(id);
    }else{
      return prev_entities_.at(id);
    }
  }
  const unordered_set<int>& next_entities(int id, bool last) const{
    if(last){
      return next_last_entities_.at(id);
    }else{
      return next_entities_.at(id);
    }
  }
};


} /* namespace coin */

#endif /* _DOCUMENT_H_ */
