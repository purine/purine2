// Copyright Lin Min 2015
#include "operations/include/bias.hpp"

namespace purine {

Bias::Bias(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
    const param_tuple& args) : Operation(inputs, outputs)  {
  Size top_size = outputs_[0]->size();
  Size bias_size = inputs_[0]->size();
  CHECK_EQ(top_size.channels(), bias_size.channels());
  CHECK_EQ(bias_size.num(), 1);
  CHECK_EQ(bias_size.height(), 1);
  CHECK_EQ(bias_size.width(), 1);
  Stride bias_stride = inputs_[0]->stride();
  cudnn::createTensor4dDesc<DTYPE>(&bias_desc_, bias_size, bias_stride);
  Stride top_stride = outputs_[0]->stride();
  cudnn::createTensor4dDesc<DTYPE>(&top_desc_, top_size, top_stride);
}

Bias::~Bias() {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
}

void Bias::compute_cpu(const vector<bool>& add) {

}

void Bias::compute_gpu(const vector<bool>& add) {
  Size s = outputs_[0]->size();
  DTYPE alpha = 1.;
  DTYPE beta = add[0] ? 1. : 0.;
  CUDNN_CHECK(cudnnAddTensor(cudnn_handle(), CUDNN_ADD_SAME_C, &alpha,
          bias_desc_, inputs_[0]->gpu_data(), &beta, top_desc_,
          outputs_[0]->mutable_gpu_data()));
}

BiasDown::BiasDown(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
  Size top_size = inputs_[0]->size();
  Size bias_size = outputs_[0]->size();
  CHECK_EQ(top_size.channels(), bias_size.channels());
  CHECK_EQ(bias_size.num(), 1);
  CHECK_EQ(bias_size.height(), 1);
  CHECK_EQ(bias_size.width(), 1);
  Stride bias_stride = outputs_[0]->stride();
  cudnn::createTensor4dDesc<DTYPE>(&bias_desc_, bias_size, bias_stride);
  Stride top_stride = inputs_[0]->stride();
  cudnn::createTensor4dDesc<DTYPE>(&top_desc_, top_size, top_stride);
}

BiasDown::~BiasDown() {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
}

void BiasDown::compute_cpu(const vector<bool>& add) {

}

void BiasDown::compute_gpu(const vector<bool>& add) {
  DTYPE alpha = 1.;
  DTYPE beta = add[0] ? 1. : 0.;
  CUDNN_CHECK(cudnnConvolutionBackwardBias(cudnn_handle(), &alpha, top_desc_,
          inputs_[0]->gpu_data(), &beta, bias_desc_,
          outputs_[0]->mutable_gpu_data()));
}

}
