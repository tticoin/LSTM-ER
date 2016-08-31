#!/bin/zsh
mkdir train
python3 conv_train.py &>! log_train &
mkdir test
python3 conv_test.py &>! log_test &
wait
java -cp ".:../common/stanford-corenlp-full-2015-04-20/*" edu.stanford.nlp.pipeline.StanfordCoreNLP -props props -outputFormat conll -filelist train_list -outputDirectory train/ &>! log_train &
java -cp ".:../common/stanford-corenlp-full-2015-04-20/*" edu.stanford.nlp.pipeline.StanfordCoreNLP -props props -outputFormat conll -filelist test_list -outputDirectory test/ &>! log_test &
wait
for i in train/*.txt
do
    perl ../common/dep2so.prl $i.conll $i > train/`basename $i .txt`.stanford.so 2> /dev/null
done &
for i in test/*.txt
do
    perl ../common/dep2so.prl $i.conll $i > test/`basename $i .txt`.stanford.so 2> /dev/null
done &
wait
mkdir corpus
mv train corpus/
mv test corpus/
