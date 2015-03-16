project(CaffeProtobuf)

cmake_minimum_required(VERSION 2.8)

find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})

PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS caffe.proto)

add_library(proto STATIC ${PROTO_SRCS} ${PROTO_HDRS})