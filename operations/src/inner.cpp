// Copyright Lin Min 2015
#include "operations/include/inner.hpp"

namespace purine {

Inner::Inner(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
    const param_tuple& args) : Operation(inputs, outputs) {
  int bottom_num = inputs_[0]->size().num();
  int top_num = outputs_[0]->size().num();
  CHECK_EQ(bottom_num, top_num);
  int bottom_channel = inputs_[0]->size().count() / inputs_[0]->size().num();
  int weight_channel = inputs_[1]->size().channels();
  int weight_num = inputs_[1]->size().num();
  int top_channel = outputs_[0]->size().channels();
  CHECK_EQ(bottom_channel, weight_channel);
  CHECK_EQ(top_channel, weight_num);
  CHECK_EQ(inputs_[1]->size().height() * inputs_[1]->size().width(), 1);
}

void Inner::compute_cpu(const vector<bool>& add) {
  Size bottom_size = inputs_[0]->size();
  Size top_size = outputs_[0]->size();
  caffe::caffe_cpu_gemm<DTYPE>(CblasNoTrans, CblasTrans, bottom_size.num(),
      top_size.channels(), bottom_size.count() / bottom_size.num(), (DTYPE)1.,
      inputs_[0]->cpu_data(), inputs_[1]->cpu_data(), add[0] ? 1. : 0.,
      outputs_[0]->mutable_cpu_data());
}

void Inner::compute_gpu(const vector<bool>& add) {
  Size bottom_size = inputs_[0]->size();
  Size top_size = outputs_[0]->size();
  caffe::caffe_gpu_gemm<DTYPE>(CblasNoTrans, CblasTrans, bottom_size.num(),
      top_size.channels(), bottom_size.count() / bottom_size.num(), (DTYPE)1.,
      inputs_[0]->gpu_data(), inputs_[1]->gpu_data(), add[0] ? 1. : 0.,
      outputs_[0]->mutable_gpu_data());
}

InnerDown::InnerDown(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
  int top_num = inputs_[0]->size().num();
  int bottom_num = outputs_[0]->size().num();
  CHECK_EQ(bottom_num, top_num);
  int bottom_channel = outputs_[0]->size().count()
      / outputs_[0]->size().num();
  int weight_channel = inputs_[1]->size().channels();
  int weight_num = inputs_[1]->size().num();
  int top_channel = inputs_[0]->size().channels();
  CHECK_EQ(bottom_channel, weight_channel);
  CHECK_EQ(top_channel, weight_num);
  CHECK_EQ(inputs_[1]->size().height() * inputs_[1]->size().width(), 1);
}

void InnerDown::compute_cpu(const vector<bool>& add) {
  Size top_size = inputs_[0]->size();
  Size bottom_size = outputs_[0]->size();
  caffe::caffe_cpu_gemm<DTYPE>(CblasNoTrans, CblasNoTrans, bottom_size.num(),
      bottom_size.count() / bottom_size.num(), top_size.channels(), (DTYPE)1.,
      inputs_[0]->cpu_data(), inputs_[1]->cpu_data(), add[0] ? 1. : 0.,
      outputs_[0]->mutable_cpu_data());
}

void InnerDown::compute_gpu(const vector<bool>& add) {
  Size top_size = inputs_[0]->size();
  Size bottom_size = outputs_[0]->size();
  caffe::caffe_gpu_gemm<DTYPE>(CblasNoTrans, CblasNoTrans, bottom_size.num(),
      bottom_size.count() / bottom_size.num(), top_size.channels(), (DTYPE)1.,
      inputs_[0]->gpu_data(), inputs_[1]->gpu_data(), add[0] ? 1. : 0.,
      outputs_[0]->mutable_gpu_data());
}

InnerWeight::InnerWeight(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
  int top_num = inputs_[0]->size().num();
  int bottom_num = inputs_[1]->size().num();
  CHECK_EQ(bottom_num, top_num);
  int bottom_channel = inputs_[1]->size().count() / inputs_[1]->size().num();
  int weight_channel = outputs_[0]->size().channels();
  int weight_num = outputs_[0]->size().num();
  int top_channel = inputs_[0]->size().channels();
  CHECK_EQ(bottom_channel, weight_channel);
  CHECK_EQ(top_channel, weight_num);
  CHECK_EQ(outputs_[0]->size().height() * outputs_[0]->size().width(), 1);
}

void InnerWeight::compute_cpu(const vector<bool>& add) {
  Size top_size = inputs_[0]->size();
  Size bottom_size = inputs_[1]->size();
  caffe::caffe_cpu_gemm<DTYPE>(CblasTrans, CblasNoTrans, top_size.channels(),
      bottom_size.count() / bottom_size.num(), bottom_size.num(), (DTYPE)1.,
      inputs_[0]->cpu_data(), inputs_[1]->cpu_data(), add[0] ? 1. : 0.,
      outputs_[0]->mutable_cpu_data());
}

void InnerWeight::compute_gpu(const vector<bool>& add) {
  Size top_size = inputs_[0]->size();
  Size bottom_size = inputs_[1]->size();
  caffe::caffe_gpu_gemm<DTYPE>(CblasTrans, CblasNoTrans, top_size.channels(),
      bottom_size.count() / bottom_size.num(), bottom_size.num(), (DTYPE)1.,
      inputs_[0]->gpu_data(), inputs_[1]->gpu_data(), add[0] ? 1. : 0.,
      outputs_[0]->mutable_gpu_data());
}

}
