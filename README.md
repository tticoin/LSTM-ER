# LSTM-ER

Implementation of [End-to-End Relation Extraction using LSTMs on Sequences and Tree Structures](http:www.aclweb.org/anthology/P/P16/P16-1105.pdf).

# Requirements

* clang++ 3.4+ (or g++ 4.8+)
* boost 1.57+
* yaml-cpp 0.5.x

For convenience, this package includes snapshot versions of clab/cnn and eigen. These follow the original license.

# Usage

* Compilation

```tar xzf cnn.tar.gz
tar xzf eigen.tar.gz
mkdir build
cd build
cmake .. -DEIGEN3_INCLUDE_DIR=eigen -DCMAKE_CXX_COMPILER=/usr/bin/clang++
make
cd ..
```

* Preparation of the pretrained embeddings

```
cd dict/
wget http://tti-coin.jp/data/wikipedia200.bin
cd ..
```

* Preparation of the data sets

see data/README.md

* Train

** ACE 2005 (Relation extraction) 
`build/relation/RelationExtraction --train -y yaml/parameter-ace2005.yaml`

** SemEval 2010 Task 8 (Relation classification) 
`build/relation/RelationExtraction --train -y yaml/parameter-semeval-2010.yaml`

* Test

** ACE 2005 (Relation extraction) 
`build/relation/RelationExtraction --test -y yaml/parameter-ace2005.yaml`

** SemEval 2010 Task 8 (Relation classification) 
`build/relation/RelationExtraction --test -y yaml/parameter-semeval-2010.yaml`

# Notes

* YAML files for ACE2004 are not included. Please modify yaml/parameter-ace2005.yaml.

* Results may not be consistent from the original paper due to the differences in the environments.

*Please refer to the following ACL paper when using this software.

* Makoto Miwa and Mohit Bansal. [End-to-end Relation Extraction using LSTMs on Sequences and Tree Structures.](http:www.aclweb.org/anthology/P/P16/P16-1105.pdf) In the Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1105--1116, 2016.
