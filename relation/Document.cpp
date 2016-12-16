/*
 * Document.cpp
 *
 *  Created on: 2014/07/05
 *      Author: miwa
 */
#include "cnn/tensor.h"
#include "Document.h"
#include "Parameter.h"
#include "Table.h"
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;
using namespace boost::filesystem;
using namespace boost::algorithm;

namespace coin {

void Dictionary::update(const Parameters& params, DocumentCollection& collection){
  unordered_map<string, unsigned> counts;
  for(Document* doc:collection.documents()){
    for(Sentence* sentence:doc->sentences()){
      for(Word* word:sentence->words()){
        if(counts.find(word->repr()) != counts.end()){
          counts[word->repr()]++;
        }else{
          counts[word->repr()] = 1;
        }
      }
    }
  }
  for(Document* doc:collection.documents()){
    for(Sentence* sentence:doc->sentences()){
      for(Word* word:sentence->words()){
        word->update(params, this, counts);
      }
    }
  }
  // add w2v words to dict
  for(pair<string, vector<float> > wv:params.w2v()){
    get_repr_id(wv.first);
  }
  for(const pair<string, int>& ent_label:ent_entry_.dict()){
    if(ent_label.first[0] == 'L' || ent_label.first[0] == 'U'){
      entity_labels_.insert(ent_label.second);
    }
  }
  for(const pair<string, int>& ent_label:ent_entry_.dict()){
    if(ent_label.first[0] == 'U'){
      begin_labels_.insert(make_pair(ent_label.second, ent_label.second));
    }else if(ent_label.first[0] == 'L'){
      for(const pair<string, int>& prev_ent_label:ent_entry_.dict()){
        if(prev_ent_label.first[0] == 'B'){
          if(prev_ent_label.first.substr(1) == ent_label.first.substr(1)){
            begin_labels_.insert(make_pair(ent_label.second, prev_ent_label.second));
            break;
          }
        }
      }
    }
  }

  for(const pair<string, int>& ent_label:ent_entry_.dict()){
    unordered_set<int> prev_labels;
    unordered_set<int> prev_first_labels;
    if(ent_label.first[0] == 'O' || ent_label.first[0] == 'U' || ent_label.first[0] == 'B'){
      for(const pair<string, int>& prev_ent_label:ent_entry_.dict()){
        if(prev_ent_label.first[0] == 'L'||
           prev_ent_label.first[0] == 'U'||
           prev_ent_label.first[0] == 'O'){
          prev_labels.insert(prev_ent_label.second);
        }
        if(prev_ent_label.first[0] == 'U'||
           prev_ent_label.first[0] == 'O'){
          prev_first_labels.insert(prev_ent_label.second);
        }        
      }
    }else{
      // ent_label.first[0] == 'L' || ent_label.first[0] == 'I'
      for(const pair<string, int>& prev_ent_label:ent_entry_.dict()){
        if(prev_ent_label.first[0] == 'I'||
           prev_ent_label.first[0] == 'B'){
          if(prev_ent_label.first.substr(1) == ent_label.first.substr(1)){
            prev_labels.insert(prev_ent_label.second);
            if(prev_ent_label.first[0] == 'B'){
              prev_first_labels.insert(prev_ent_label.second);
            }
          }
        }
      }
    }
    prev_entities_.insert(make_pair(ent_label.second, prev_labels));
    prev_first_entities_.insert(make_pair(ent_label.second, prev_first_labels));
  }
  for(const pair<string, int>& ent_label:ent_entry_.dict()){
    unordered_set<int> next_labels;
    unordered_set<int> next_last_labels;
    if(ent_label.first[0] == 'O' || ent_label.first[0] == 'U' || ent_label.first[0] == 'L'){
      for(const pair<string, int>& next_ent_label:ent_entry_.dict()){
        if(next_ent_label.first[0] == 'B'||
           next_ent_label.first[0] == 'U'||
           next_ent_label.first[0] == 'O'){
          next_labels.insert(next_ent_label.second);
        }
        if(next_ent_label.first[0] == 'U'||
           next_ent_label.first[0] == 'O'){
          next_last_labels.insert(next_ent_label.second);
        }
      }
    }else{
      // ent_label.first[0] == 'B' || ent_label.first[0] == 'I'
      for(const pair<string, int>& next_ent_label:ent_entry_.dict()){
        if(next_ent_label.first[0] == 'I'||
           next_ent_label.first[0] == 'L'){
          if(next_ent_label.first.substr(1) == ent_label.first.substr(1)){
            next_labels.insert(next_ent_label.second);
            if(next_ent_label.first[0] == 'L'){
              next_last_labels.insert(next_ent_label.second);
            }
          }
        }
      }
    }
    next_entities_.insert(make_pair(ent_label.second, next_labels));
    next_last_entities_.insert(make_pair(ent_label.second, next_last_labels));
  }
  for(Document* doc:collection.documents()){
    for(Sentence* sentence:doc->sentences()){
      sentence->update_relations(this);
      sentence->calculate_shortest_paths();
      sentence->calculate_dep_tree(this);
    }
  }
}

void Dictionary::apply(Document& doc) const{
  for(Sentence* sentence:doc.sentences()){
    for(Word* word:sentence->words()){
      word->apply(this, sentence);
    }
    sentence->apply_relations(this);
    sentence->calculate_shortest_paths();
    sentence->calculate_dep_tree(this);
  }
}

void Dictionary::apply(DocumentCollection& collection) const{
  for(Document* doc:collection.documents()){
    apply(*doc);
  }
}


void Word::update(const Parameters& params, Dictionary* dict, unordered_map<string, unsigned>& counts){
  if(counts[repr_] < params.min_frequency() && params.w2v().find(repr_) == params.w2v().end()){
    if(cnn::rand01() < params.unk_prob()){
      // ignore this word
      repr_ = "UNK";
      pos_ = "UNK";
    }
  }
  repr_id_ = dict->get_repr_id(repr_);
  pos_id_ = dict->get_pos_id(pos_);
  wn_id_ = dict->get_wn_id(wn_);
  label_id_ = dict->get_ent_id(label_);
  for(pair<string, unordered_set<Word*> > dep:dependencies_){
    int id = 0;
    if(dep.first.find_first_of(REVERSE_DEP_HEADER) == 0){
      string fwd = dep.first.substr(REVERSE_DEP_HEADER.size());
      dict->get_dep_id(fwd);
      string rev = REVERSE_DEP_HEADER + fwd;
      id = dict->get_dep_id(rev);
    }else{
      string fwd = dep.first;
      id = dict->get_dep_id(fwd);
      string rev = REVERSE_DEP_HEADER + fwd;
      dict->get_dep_id(rev);
      if(dict->get_dep_id(rev) != id+1){
        cerr << "reverse relation is set beforehand" << endl;
      }
      assert(dict->get_dep_id(rev) == id+1);
    }
    dependencies_ids_.insert(make_pair(id, dep.second));
  }
}

void Word::apply(const Dictionary* dict, Sentence *sentence){
  if(dict->get_repr_id(repr_) < 0){
    // ignore this word
    repr_ = "UNK";
  }
  repr_id_ = dict->get_repr_id(repr_);
  if(repr_id_ < 0){
    cerr << "no conversion for " << repr_ << endl;
  }
  assert(repr_id_ >= 0);
  pos_id_ = dict->get_pos_id(pos_);
  if(pos_id_ < 0){
    pos_ = "UNK";
    pos_id_ = dict->get_pos_id(pos_);
  }
  wn_id_ = dict->get_wn_id(wn_);
  label_id_ = dict->get_ent_id(label_);
  if(label_id_ < 0){
    cerr << "unknown label:" << label_ << ", treated as negative " << endl;
    if(label_[0] == 'L' || label_[0] == 'U'){
      sentence->inc_missing_terms();
    }
    label_id_ = NEGATIVE_ENTITY_ID;
  }
  for(pair<string, unordered_set<Word*> > dep:dependencies_){
    int id = 0;
    if(dep.first.find_first_of(REVERSE_DEP_HEADER) == 0){
      string fwd = dep.first.substr(REVERSE_DEP_HEADER.size());
      string rev = REVERSE_DEP_HEADER + fwd;
      id = dict->get_dep_id(rev);
    }else{
      string fwd = dep.first;
      id = dict->get_dep_id(fwd);
    }
    if(id == -1 || id == 0){
      cerr << "unknown dependency label:" << dep.first << endl;
      // TODO: consider unknown dependency??
      id = NEGATIVE_DEPENDENCY_ID;
    }
    dependencies_ids_.insert(make_pair(id, dep.second));
  }
}

DocumentCollection::DocumentCollection(const Parameters& params, const string& dir){
  if(VERBOSITY > 1){
    cerr << "start loading documents in " << dir << endl;
  }
  path p(dir.c_str());
  if(exists(p) && is_directory(p)){
    for(directory_iterator it(p); it != directory_iterator(); ++it){
      if(is_regular_file(*it) && ends_with(it->path().string(), params.text_ext())){
        Document* doc = new Document(params, it->path().string().substr(0, it->path().string().length() - params.text_ext().length()));
        add_document(doc);
      }
    }
  }
}

DocumentCollection::~DocumentCollection() {
  for(const Document *doc:documents_){
    delete doc;
  }
}

void DocumentCollection::add_document(Document* doc){
  documents_.push_back(doc);
}

vector<Table*> DocumentCollection::collect_tables(){
  vector<Table*> tables;
  for(Document *document:documents_){
    vector<Table*> &doc_tables = document->tables();
    tables.insert(tables.end(), doc_tables.begin(), doc_tables.end());
  }
  return tables;
}

Document::Document(const Parameters& params, const string& base):params_(params), has_annotation_(false), id_(base) {
  read_text(base+params.text_ext());
  read_parse(base+params.parsers().at(params.base_parser()).extension(), params.parsers().at(params.base_parser()));
  read_annotation(base+params.ann_ext());
}

vector<Table*>& Document::tables(){
  if(tables_.size() == 0){
    for(Sentence *sentence:sentences_){
      tables_.push_back(new Table(*sentence));
    }
  }
  return tables_;
}

Document::~Document(){
  for(Table* table:tables_){
    delete table;
  }
  for(Sentence* sentence:sentences_){
    delete sentence;
  }
  for(pair<string, Term*> term:terms_){
    delete term.second;
  }
  for(pair<string, Relation*> relation:relations_){
    delete relation.second;
  }
  delete text_;
};

UnicodeString* Document::read_file(const string& file) const{
  std::ifstream ifs;
  ifs.open(file, ios::binary);
  if(!ifs){
    if(VERBOSITY > 0){
      cerr << "Cannot open " << file << endl;
    }
    exit(-1);
  }
  long int len;
  ifs.seekg(0, ios::end);
  len = ifs.tellg();
  ifs.seekg(0, ios::beg);
  char *tmp = new char[len+1];
  ifs.read(tmp, len);
  UnicodeString *text = new UnicodeString(tmp, "utf8");
  delete[] tmp;
  return text;
}

string Document::convert(const UnicodeString& str) const{
  char result[str.length()*2];
  str.extract(0, str.length(), result, "utf8");
  return string(result);
}

unordered_map<string, string> parse_attributes(const string& attr_string){
  //TODO: multiple attributes
  unordered_map<string, string> attrs;
  size_t current_index = 0, index;
  string key = "", value = "";
  while((index = attr_string.find_first_of('"', current_index)) != string::npos){
    if(key == "" && attr_string[index-1] == '='){
      while(attr_string[current_index] == ' ')current_index++;
      key = attr_string.substr(current_index, index-1-current_index);
    }else if(attr_string[index-1] == '\\'){
      value += attr_string.substr(current_index, index+1-current_index);
    }else{
      value += attr_string.substr(current_index, index-current_index);
      attrs[key] = value;
      key = "";
      value = "";
    }
    current_index = index + 1;
  }
  return attrs;
}

void Document::read_parse(const string& file, const ParseParameters& parse) {
  std::ifstream ifs;
  ifs.open(file);
  if(!ifs){
    if(VERBOSITY > 0){
      cerr << "Cannot open " << file << endl;
    }
    exit(-1);
  }
  string line;
  int current_sentence_idx = 0;
  Sentence *current_sentence = nullptr;
  if(!parse.is_base()){
    current_sentence = sentences_[current_sentence_idx];
  }
  while(getline(ifs, line)){
    if(line == "")continue;
    vector<string> annotations;
    split(annotations, line, bind2nd(equal_to<char>(), '\t'));
    int start = atoi(annotations[0].c_str());
    int end = atoi(annotations[1].c_str());
    string tag;
    string attr_string;
    if(annotations.size() > 3){
      tag = annotations[2];
      attr_string = annotations[3];
    }else{
      size_t split = annotations[2].find_first_of(' ', 0);
      if(split == string::npos){
        tag = annotations[2];
        attr_string = "";
      }else{
        tag = annotations[2].substr(0, split);
        attr_string = annotations[2].substr(split+1, annotations[2].size() - split - 1);
      }
    }
    unordered_map<string, string> attrs = parse_attributes(attr_string);
    if(tag == parse.sentence_tag()){
      if(parse.is_base()){
        Sentence *sentence = new Sentence(start, end, attrs[parse.id_attr()], *this);
        sentences_.push_back(sentence);
        current_sentence = sentence;
      }
    }else{
      if(!parse.is_base()){
        while(start > current_sentence->end()){
          current_sentence = sentences_[++current_sentence_idx];
        }
      }
      if(tag == parse.token_tag()){
        Word *word = new Word(params_, start, end, text(start, end - start), attrs, parse);
        if(parse.is_base()){
          if(current_sentence == nullptr){
            cerr << "no sentence for word (" << start << "," << end << ")" << endl;
          }
          assert(current_sentence != nullptr);
          current_sentence->add(word);
        }
        current_sentence->add(word, parse.type());
      }else{
        Constituent *cons = new Constituent(start, end, attrs, parse);
        current_sentence->add(cons, parse.type());
      }
    }
  }
  for(Sentence *sentence:sentences_){
    sentence->build_dependencies(parse);
  }
}

bool term_less(Term*& t1, Term*& t2){
  if (t1->start() < t2->start())return true;
  if (t1->start() > t2->start())return false;
  if (t1->end() < t2->end())return false;
  if (t1->end() > t2->end())return true;
  return t1->id() > t2->id();
}


void Document::read_annotation(const string& file) {
  if(sentences_.size() == 0){
    cerr << "document has no sentence" << endl;
  }
  assert(sentences_.size() > 0);
  std::ifstream ifs;
  ifs.open(file);
  if(ifs){
    has_annotation_ = true;
    string line;
    while(getline(ifs, line)){
      if(line[0] == 'T'){
        string id, type;
        int start, end;
        vector<string> annotations;
        split(annotations, line, bind2nd(equal_to<char>(), '\t'));
        id = annotations[0];
        istringstream ss(annotations[1]);
        ss >> type >> start >> end;
        terms_[id] = new Term(start, end, id, type);
      }else if(line[0] == 'R'){
        string id, type, arg1, arg2;
        vector<string> annotations;
        split(annotations, line, bind2nd(equal_to<char>(), '\t'));
        id = annotations[0];
        istringstream ss(annotations[1]);
        ss >> type >> arg1 >> arg2;
        int s1 = arg1.find_first_of(':', 0);
        int s2 = arg2.find_first_of(':', 0);
        string h1 = arg1.substr(0, s1);
        string h2 = arg2.substr(0, s2);
        if(h1 < h2){
          type = h1 + ":" + type + ":" + h2;
          arg1 = arg1.substr(s1+1);
          arg2 = arg2.substr(s2+1);
        }else{
          type = h2 + ":" + type + ":"+ h1;
          string tmp = arg1;
          arg1 = arg2.substr(s2+1);
          arg2 = tmp.substr(s1+1);
        }
        relations_[id] = new Relation(id, type, arg1, arg2);
      }
    }
    // influence annotation
    list<Term*> terms;
    for(pair<string, Term*> term:terms_){
      terms.push_back(term.second);
    }
    terms.sort(term_less);
    list<Term*>::iterator term_it = terms.begin();
    list<Term*>::iterator term_end = terms.end();

    for(Sentence* sentence:sentences_){
      while(term_it != term_end && (*term_it)->start() < sentence->start() && sentence->start() < (*term_it)->end()){
        cerr << "Term " << sentence->doc().id() << ":" << (*term_it)->id() << " is intersentential. Ignored." << endl;
        sentence->inc_missing_terms();
        ++term_it;
      }
      // add annotation to sentence
      while(term_it != term_end && (*term_it)->inside_of(*sentence)){
        sentence->add(*term_it);
        ++term_it;
      }   
      for(pair<string, Relation*> rel:relations_){
        if(sentence->contains_term(rel.second->arg1()) &&
           sentence->contains_term(rel.second->arg2())){
          sentence->add(rel.second);
        }else if(sentence->contains_term(rel.second->arg1())){
          cerr << "Relation " << sentence->doc().id() << ":" << rel.second->id() << " is intersentential. Ignored." << endl;
          sentence->inc_missing_rels();
        }
      }
      sentence->build_word_annotations();
    }
    ifs.close();
  }else{
    for(Sentence* sentence:sentences_){
      sentence->build_word_annotations();
    }
  }
}


IndexRelation::IndexRelation(const string& id, int arg1, int arg2, const string& type, Dictionary *dict):id_(id){
  if(arg1 == arg2){
    cerr << "self-relation is detected for " << arg1 << endl;
  }
  assert(arg1 != arg2);
  if(arg1 < arg2){
    arg1_ = arg1;
    arg2_ = arg2;
    type_  = dict->get_rel_id(type);
    if(type != NEGATIVE_RELATION){
      dict->get_rel_id(REVERSE_RELATION_HEADER+type);
    }
  }else{
    arg1_ = arg2;
    arg2_ = arg1;
    if(type != NEGATIVE_RELATION){
      dict->get_rel_id(type);
      type_  = dict->get_rel_id(REVERSE_RELATION_HEADER+type);
    }else{
      type_  = dict->get_rel_id(type);
    }
  }
}

IndexRelation::IndexRelation(const string& id, int arg1, int arg2, const string& type, const Dictionary *dict):id_(id){
  assert(arg1 != arg2);
  if(arg1 < arg2){
    arg1_ = arg1;
    arg2_ = arg2;
    type_  = dict->get_rel_id(type);
  }else{
    arg1_ = arg2;
    arg2_ = arg1;
    if(type != NEGATIVE_RELATION){
      type_  = dict->get_rel_id(REVERSE_RELATION_HEADER+type);
    }else{
      type_  = dict->get_rel_id(type);
    }
  }
  if(type_ < 0){
    cerr << "unknown type:" << type << endl;
    type_ = NEGATIVE_RELATION_ID;
  }
}

void Sentence::build_dependencies(const ParseParameters& parse){
  unordered_map<string, Constituent*>& parse_nodes = nodes_[parse.type()];
  // set head
  string head_attr = parse.head_attr();
  if(head_attr != ""){
    for(pair<string, Constituent*> parse_node:parse_nodes){
      Constituent* cons = parse_node.second;
      if(cons->attrs().find(head_attr) != cons->attrs().end()){
        cons->set_head(parse_nodes.at(cons->attrs().at(head_attr)));
      }
    }
  }
  // add dependant
  for(pair<string, Constituent*> parse_node:parse_nodes){
    if(parse_node.second->head() == nullptr){
      // word
      Word *word = (Word*)parse_node.second;
      string pred = "";
      if(parse.pred_attr() != ""){
        if(word->attrs().find(parse.pred_attr()) == word->attrs().end()){
          // no dependency
          continue;
        }
        pred = word->attrs().at(parse.pred_attr());
      }
      for(pair<string, string> attr:word->attrs()){
        if(attr.first == parse.base_attr())continue;
        if(attr.first == parse.head_attr())continue;
        if(attr.first == parse.id_attr())continue;
        if(attr.first == parse.pos_attr())continue;
        if(attr.first == parse.wn_attr())continue;
        if(attr.first == parse.pred_attr())continue;
        if(parse_nodes.find(attr.second) != parse_nodes.end()){
          // dependency found!!
          Constituent *dependant = parse_nodes.at(attr.second);
          while(dependant->head() != nullptr){
            dependant = dependant->head();
          }
          if(((Word*)dependant) == word){
            continue;
          }
          string fwd = pred+attr.first;
          // if(fwd == "aux" || fwd == "auxpass"){
          //   fwd = "aux";
          // }else if(fwd == "acomp" || fwd == "ccomp" || fwd == "xcomp" || fwd == "pcomp"){
          //   fwd = "comp";
          // }else if(fwd == "amod" || fwd == "advcl" || fwd == "advmod" || fwd == "appos" ||
          //          fwd == "det" || fwd == "mark" || fwd == "mwe" || fwd == "neg" ||
          //          fwd == "nn" || fwd == "npadvmod" || fwd == "num" || fwd == "number" ||
          //          fwd == "poss" || fwd == "possessive" || fwd == "preconj" || fwd == "predet" ||
          //          fwd == "prep" || fwd == "prt" || fwd == "quantmod" || fwd == "rcmod" ||
          //          fwd == "tmod" || fwd == "vmod"){
          //   fwd = "mod";
          // }
          word->set_dependency(fwd, (Word*)dependant);
          string rev = REVERSE_DEP_HEADER+fwd;
          ((Word*)dependant)->set_dependency(rev, word);
        }
      }
    }
  }
}

void Sentence::build_word_annotations(){
  int nwords = words_.size();
  vector<Term*>::iterator term_end = terms_.end();
  for(vector<Term*>::iterator term_it = terms_.begin();term_it != term_end;++term_it){
    for(int idx = 0;idx < nwords;++idx){
      Word* word = words_[idx];
      if((*term_it)->overlap(*word)){
        (*term_it)->add(word);
        word->add(*term_it);
      }
    }
  }
  for(Word *word:words_){
    if(word->terms().size() > 1){
      for(vector<Term*>::iterator term_it = word->terms().begin();
          term_it != word->terms().end();){
        if((*term_it)->words().size() == 1){
          ++term_it;
          continue;
        }
        if(!(*term_it)->is_edge(word)){
          ++term_it;
          continue;
        }
        (*term_it)->remove(word);
        term_it = word->terms().erase(term_it);
      }
    }
    if(word->terms().size() > 1){
      int count = -1;
      Term* to_keep = nullptr;
      for(vector<Term*>::iterator term_it = word->terms().begin();
          term_it != word->terms().end();++term_it){
        int rel_counts = term_rel_counts_[(*term_it)->id()];
        if(rel_counts > count){
          count = rel_counts;
          to_keep = *term_it;
        }
      }    
      assert(to_keep != nullptr);
      for(vector<Term*>::iterator term_it = word->terms().begin();
          term_it != word->terms().end();){
        if(*term_it == to_keep){
          ++term_it;
        }else{
          (*term_it)->remove(word);
          term_it = word->terms().erase(term_it);
        }
      }
    }
  }
  for(Term* term:terms_){
    int term_words = term->words().size();
    if(term_words == 0){
      cerr << "Term " << doc().id() << ":" << term->id() << " is overlapping. Ignored." << endl;
      inc_missing_terms();
      continue;
    }
    assert(term_words > 0);
    if(term_words == 1){
      term->words()[0]->set_label("U-"+term->type());
      term->words()[0]->set_direct_term(term);
    }else{
      term->words()[0]->set_label("B-"+term->type());
      for(int i = 1;i < term_words-1;++i){
        term->words()[i]->set_label("I-"+term->type());
      }
      term->words()[term_words-1]->set_label("L-"+term->type());
      term->words()[term_words-1]->set_direct_term(term);
    }
  }
  for(int idx = 0;idx < nwords;++idx){
    Word *word = words_[idx];
    if(word->label() == ""){
      word->set_label("O");
    }
  }
}

void Sentence::update_relations(Dictionary *dict){
  int nwords = words_.size();
  unordered_map<string, int> term_to_widx;
  for(int idx = 0;idx < nwords;++idx){
    Word *word = words_[idx];
    if(word->direct_term() != nullptr){
      term_to_widx[word->direct_term()->id()] = idx;
    }
  }
  unordered_map<string, int>::iterator end_it = term_to_widx.end();
  unordered_set<int> covered;
  for(Relation* rel:rels_){
    if(term_to_widx.find(rel->arg1()) != end_it &&
       term_to_widx.find(rel->arg2()) != end_it){
      assert(dict->is_entity_label(words_[term_to_widx.at(rel->arg1())]->label_id()));
      assert(dict->is_entity_label(words_[term_to_widx.at(rel->arg2())]->label_id()));
      int arg1 = term_to_widx[rel->arg1()], arg2 = term_to_widx[rel->arg2()];
      int uniq_id12 = arg1 * nwords + arg2, uniq_id21 = arg2 * nwords + arg1;
      if((covered.find(uniq_id12) != covered.end()) ||
         (covered.find(uniq_id21) != covered.end())){
        cerr << "Relation " << doc().id() << ":" << rel->id() << " is overlapping. Ignored." << endl;
        inc_missing_rels();
        continue;
      }
      covered.insert(uniq_id12);
      covered.insert(uniq_id21);
      index_rels_.push_back(IndexRelation(rel->id(), arg1, arg2, rel->type(), dict));
    }else{
      cerr << "Relation " << doc().id() << ":" << rel->id() << " misses entity. Ignored." << endl;
      inc_missing_rels();
    }
  }
}

void Sentence::apply_relations(const Dictionary *dict){
  int nwords = words_.size();
  unordered_map<string, int> term_to_widx;
  for(int idx = 0;idx < nwords;++idx){
    Word *word = words_[idx];
    if(word->direct_term() != nullptr){
      term_to_widx[word->direct_term()->id()] = idx;
    }
  }
  unordered_set<int> covered;
  unordered_map<string, int>::iterator end_it = term_to_widx.end();
  for(Relation* rel:rels_){
    if(term_to_widx.find(rel->arg1()) != end_it && term_to_widx.find(rel->arg2()) != end_it){
      int arg1 = term_to_widx[rel->arg1()], arg2 = term_to_widx[rel->arg2()];
      int uniq_id12 = arg1 * nwords + arg2, uniq_id21 = arg2 * nwords + arg1;
      if((covered.find(uniq_id12) != covered.end()) ||
         (covered.find(uniq_id21) != covered.end())){
        cerr << "Relation " << doc().id() << ":" << rel->id() << " is overlapping. Ignored." << endl;
        inc_missing_rels();
        continue;
      }
      if(dict->get_rel_id(rel->type()) == -1){
        inc_missing_rels();
      }      
      covered.insert(uniq_id12);
      covered.insert(uniq_id21);
      index_rels_.push_back(IndexRelation(rel->id(), term_to_widx[rel->arg1()], term_to_widx[rel->arg2()], rel->type(), dict));
    }
  }
}

Sentence::~Sentence(){
  for(pair<string, unordered_map<string, Constituent*>> parse_nodes:nodes_){
    for(pair<string, Constituent*> node:parse_nodes.second){
      delete node.second;
    }
  }
}

void Sentence::calculate_shortest_paths(){
  unordered_map<uint64_t, int> deps;
  unordered_map<Word*, int> word_idx;
  
  const int nwords = words_.size();
  vector<vector<int>> dist;
  vector<vector<int>> next;
  for(int i = 0;i < nwords;++i){
    Word *word = words_[i];
    word_idx[word] = i;        
    dist.push_back(vector<int>());
    next.push_back(vector<int>());
    for(int j = 0;j < nwords;++j){
      dist[i].push_back(-1);
      next[i].push_back(-1);
    }
  }
  for(int idx = 0;idx < nwords;++idx){
    for(pair<int, unordered_set<Word*> > dep:words_[idx]->dependencies_ids()){
      for(Word* word:dep.second){
        int target = word_idx[word];
        uint64_t key = idx;
        key = key << 32 | target;
        deps.insert(make_pair(key, dep.first));
        assert(idx != target);
        dist[idx][target] = 1;
        next[idx][target] = target;
      }
    }
  }
  // for(int idx = 0;idx < nwords;++idx){
  //   for(pair<int, unordered_set<Word*> > dep:words_[idx]->dependencies_ids()){
  //     for(Word* word:dep.second){
  //       int target = word_idx[word];
  //       assert(dist[target][idx] == 1);
  //       assert(next[target][idx] == idx);
  //     }
  //   }
  // }
  // Floyd-Warshall
  for(int k = 0;k < nwords;++k){
    for(int i = 0;i < nwords;++i){
      for(int j = 0;j < nwords;++j){
        if(dist[i][k] == -1 || dist[k][j] == -1)continue;
        if(dist[i][j] == -1 || dist[i][k] + dist[k][j] < dist[i][j]){
          dist[i][j] = dist[i][k] + dist[k][j];
          assert(next[i][k] != -1);
          next[i][j] = next[i][k];
        }
      }
    }
  }
  for(int i = 0;i < nwords;++i){
    paths_.push_back(vector<pair<int, int> >());
    for(int j = 0;j < nwords;++j){
      if(next[i][j] != -1){
        assert(next[j][i] != -1);
        uint64_t key = i;
        key = key << 32 | next[i][j];
        assert(deps.find(key) != deps.end());
        paths_[i].push_back(make_pair(next[i][j], deps[key]));
      }else{
        paths_[i].push_back(make_pair(-1, -1));
      }
    }
  }
}

deque<int> Sentence::get_path(int from, int to) const{
  if(paths_[from][to].first == -1){
    return deque<int>();
  }
  deque<int> sp;
  sp.push_back(from);
  while(from != to){
    sp.push_back(paths_[from][to].second);
    from = paths_[from][to].first;
    sp.push_back(from);
  }
  return sp;
}

void Sentence::calculate_dep_tree(const Dictionary* dict){
  unordered_map<Word*, int> word_idx;
  const int nwords = words_.size();
  for(int idx = 0;idx < nwords;++idx){
    word_idx[words_[idx]] = idx;
    TreeNode *node = new TreeNode(idx);
    dep_tree_.add_node(node); // ordered
  }
  for(int idx = 0;idx < nwords;++idx){
    for(pair<int, unordered_set<Word*> > dep:words_[idx]->dependencies_ids()){
      if(dict->get_dep_string(dep.first).find_first_of(REVERSE_DEP_HEADER) == 0)continue;
      for(Word* word:dep.second){
        int target = word_idx[word];
        dep_tree_.node(idx)->set_parent(target);
        dep_tree_.node(idx)->set_dep(dep.first);
        dep_tree_.node(target)->add_child(idx);
      }      
    }
  }
  //int max_child = 0;
  for(int idx = 0;idx < nwords;++idx){
    if(dep_tree_.node(idx)->parent() < 0){
      dep_tree_.set_root(dep_tree_.node(idx));
    }
    // if(max_child < dep_tree_.node(idx)->children().size()){
    //    max_child = dep_tree_.node(idx)->children().size();
    // }
  }
  //if(dep_tree_.root() != nullptr){
  //  cerr << "id:" << doc().id() << ":" << id_ << endl;
  //}
  assert(dep_tree_.root() != nullptr);
}

} /* namespace coin */
