# Requirements

* python3
* perl
* nltk (for stanford pos tagger)
* java (for stanford tools)
* zsh

# Usage

## download Stanford Core NLP & POS tagger

```
cd common
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2015-04-20.zip
wget http://nlp.stanford.edu/software/stanford-postagger-2015-04-20.zip
unzip stanford-corenlp-full-2015-04-20.zip
unzip stanford-postagger-2015-04-20.zip
cd ..
```

## copy and convert each corpus (Please replace the directories for the original corpora.)

### ACE 2004

```
cp -r ${ACE2004_DIR}/*/english ace2004/
cd ace2004
zsh run.zsh
cd ..
```

### ACE 2005

```
cp -r ${ACE2005_DIR}/*/English ace2005/
cd ace2005
zsh run.zsh
cd ..
```

### SemEval 2010 Task 8

```
cp ${SEMEVAL_TRAIN_DIR}/TRAIN_FILE.TXT semeval-2010/
cp ${SEMEVAL_TEST_DIR}/TEST_FILE.txt semeval-2010/
cd semeval-2010/
zsh run.zsh
cd ..
```
