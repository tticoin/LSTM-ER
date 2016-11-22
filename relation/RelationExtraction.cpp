/*
 * RelationExtraction.cpp
 *
 *  Created on: 2015/09/13
 *      Author: miwa
 */
#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include "Parameter.h"
#include "Document.h"
#include "RelLstmModel.h"

using namespace std;
using namespace coin;
using namespace boost::program_options; 
using namespace boost::filesystem;
using namespace boost::algorithm;

int main(int argc, char **argv){
  cnn::Initialize(argc, argv, 1);
  options_description options("Usage");
  options.add_options()
    ("help,h", "help")
    ("test", "test")
    ("train", "train")
    ("yaml,y", value<string>()->required(), "yaml file");
  variables_map variables;
  try{
    store(parse_command_line(argc, argv, options), variables);
  }catch(exception &e){
    cerr << e.what() << endl;
    exit(-1);
  }
  bool do_train = false;
  bool do_test = false;
  if(variables.count("train")){
    do_train = true;
  }
  if(variables.count("test")){
    do_test = true;
  }
  if(variables.count("help") || !(do_train ^ do_test)){
    cout << options << endl;
    return 0;
  } 
  string yaml_file = variables["yaml"].as<string>();
  Parameters params(yaml_file);
  if(do_train){
    DocumentCollection train_data(params, params.train_dir());
    Dictionary dict;
    dict.update(params, train_data);
    vector<Table*> train_tables = train_data.collect_tables();
    DocumentCollection test_data(params, params.test_dir());
    dict.apply(test_data);
    vector<Table*> test_tables = test_data.collect_tables();
    
    if(params.do_ner() && params.entity_iteration() > 0){
      // pretraining
      RelLSTMModel model(params, dict);
      for(int i = 1;i <= params.entity_iteration();++i){
        cerr << "==== iteration: " << i << " =====" << endl;
        cerr << "train: " << endl;
        model.update(train_tables, true);
        cerr << "test: " << endl;
        model.predict(test_tables, true);
      }
      std::ofstream os(params.model_file());
      model.save_model(os);
      os.close();
    }
    RelLSTMModel *model;
    Dictionary dict2;
    if(params.do_ner() && params.entity_iteration() > 0){
      // pretraining
      std::ifstream is(params.model_file());
      model = new RelLSTMModel(params, dict2, is);
      is.close();
    }else{
      // no pretraining
      model = new RelLSTMModel(params, dict);
    }
    for(int i = 1;i <= params.iteration();++i){
      cerr << "==== iteration: " << i << " =====" << endl;
      cerr << "train: " << endl;
      model->update(train_tables);
      cerr << "test: " << endl;
      model->predict(test_tables);
    }
    std::ofstream os(params.model_file());
    model->save_model(os);
    os.close();
    delete model;
  }else if(do_test){
    Dictionary dict;
    std::ifstream is(params.model_file());
    RelLSTMModel model(params, dict, is);
    is.close();
    path p(params.test_dir().c_str());
    if(exists(p) && is_directory(p)){
      for(directory_iterator it(p); it != directory_iterator(); ++it){
        if(is_regular_file(*it) && ends_with(it->path().string(), params.text_ext())){
          Document doc(params, it->path().string().substr(0, it->path().string().length() - params.text_ext().length()));
          dict.apply(doc);
          model.output(doc.tables());
        }
      }
    }
  }
  return 0;
}



