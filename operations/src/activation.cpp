// Copyright Lin Min 2015
#include "operations/include/activation.hpp"

namespace purine {

Activation::Activation(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
  std::tie(mode_) = args;
  CHECK_EQ(inputs_[0]->size(), outputs_[0]->size());
  Size bottom_size = inputs_[0]->size();
  Stride bottom_stride = inputs_[0]->stride();
  Size top_size = outputs_[0]->size();
  Stride top_stride = outputs_[0]->stride();
  cudnn::createTensor4dDesc<DTYPE>(&bottom_desc_, bottom_size, bottom_stride);
  cudnn::createTensor4dDesc<DTYPE>(&top_desc_, top_size, top_stride);
  if (mode_ == "relu") {
    activation_mode_ = CUDNN_ACTIVATION_RELU;
  } else if (mode_ == "sigmoid") {
    activation_mode_ = CUDNN_ACTIVATION_SIGMOID;
  } else if (mode_ == "tanh") {
    activation_mode_ = CUDNN_ACTIVATION_TANH;
  } else {
    LOG(FATAL) << "Unknown activation mode " << mode_;
  }
}

Activation::~Activation() {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
}

void Activation::compute_gpu(const vector<bool>& add) {
  DTYPE alpha = 1.;
  DTYPE beta = add[0] ? 1. : 0.;
  CUDNN_CHECK(cudnnActivationForward(cudnn_handle(), activation_mode_,
          &alpha, bottom_desc_, inputs_[0]->gpu_data(), &beta, top_desc_,
          outputs_[0]->mutable_gpu_data()));
}

ActivationDown::ActivationDown(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
  std::tie(mode_) = args;
  CHECK_EQ(inputs_[0]->size(), outputs_[0]->size());
  Size bottom_size = outputs_[0]->size();
  Stride bottom_stride = outputs_[0]->stride();
  Size top_size = inputs_[0]->size();
  Stride top_stride = inputs_[0]->stride();
  cudnn::createTensor4dDesc<DTYPE>(&bottom_desc_, bottom_size, bottom_stride);
  cudnn::createTensor4dDesc<DTYPE>(&top_desc_, top_size, top_stride);
  if (mode_ == "relu") {
    activation_mode_ = CUDNN_ACTIVATION_RELU;
  } else if (mode_ == "sigmoid") {
    activation_mode_ = CUDNN_ACTIVATION_SIGMOID;
  } else if (mode_ == "tanh") {
    activation_mode_ = CUDNN_ACTIVATION_TANH;
  } else {
    LOG(FATAL) << "Unknown activation mode " << mode_;
  }
}

ActivationDown::~ActivationDown() {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
}

void ActivationDown::compute_gpu(const vector<bool>& add) {
  DTYPE alpha = 1.;
  DTYPE beta = add[0] ? 1. : 0.;
  CUDNN_CHECK(cudnnActivationBackward(cudnn_handle(), activation_mode_,
          &alpha, top_desc_, inputs_[1]->gpu_data(), top_desc_,
          inputs_[0]->gpu_data(), bottom_desc_, inputs_[2]->gpu_data(),
          &beta, bottom_desc_, outputs_[0]->mutable_gpu_data()));
}

}
