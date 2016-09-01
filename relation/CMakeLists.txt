PROJECT(cnn:relation)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

ADD_EXECUTABLE(RelationExtraction RelationExtraction.cpp Document.cpp Parameter.cpp RelLstmModel.cpp Table.cpp cnn/deptreelstm.cc cnn/zlstm.cc)
target_link_libraries(RelationExtraction cnn yaml-cpp icuuc boost_filesystem boost_system ${Boost_LIBRARIES})

