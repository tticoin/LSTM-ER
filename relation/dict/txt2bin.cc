#include <cstdlib>
#include <cstdio>
#include <unordered_map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cassert>
using namespace std;

const long long max_w = 50;              // max length of vocabulary entries
const long long dimension = 50;
int main(int argc, char *argv[]){
  ifstream is(argv[1]);
  unordered_map<string, vector<float> > vectors;
  while(is){
    string s;
    is >> s;
    if(s == "")break;
    vector<float> vec;
    for(int i = 0;i < dimension;++i){
      float v;
      is >> v;
      assert(v < 100.);
      vec.push_back(v);
    }
    if(s.size() >= max_w)continue;
    vectors.insert(make_pair(s, vec));
  }
  FILE *f;
  f = fopen(argv[2], "wb");
  long long words = vectors.size(), size = dimension;
  cerr << words << " words, " << size << " dimensions" << endl;
  fprintf(f, "%lld ", words);
  fprintf(f, "%lld ", size);

  for(pair<string, vector<float> > wv:vectors){
    fprintf(f, "%s ", wv.first.c_str());
    assert(wv.second.size() == dimension);
    for (float v:wv.second){
      assert(v < 100.);
      fwrite(&v, sizeof(float), 1, f);
    }
  }
  fclose(f);
  return 0;
}
