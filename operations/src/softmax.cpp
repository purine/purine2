// Copyright Lin Min 2015
#include <string>
#include <cfloat>

#include "operations/include/softmax.hpp"
#include "caffeine/cudnn.hpp"
#include "caffeine/math_functions.hpp"

using std::string;

namespace purine {

Softmax::Softmax(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
  std::tie(mode) = args;
  CHECK_EQ(inputs_[0]->size(), outputs_[0]->size());
  Size bottom_size = inputs_[0]->size();
  Size top_size = outputs_[0]->size();
  Stride bottom_stride = inputs_[0]->stride();
  Stride top_stride = outputs_[0]->stride();
  cudnn::createTensor4dDesc<DTYPE>(&bottom_desc_, bottom_size.num(),
      bottom_size.channels(), bottom_size.height(), bottom_size.width(),
      bottom_stride.nstride(), bottom_stride.cstride(),
      bottom_stride.hstride(), bottom_stride.wstride());
  cudnn::createTensor4dDesc<DTYPE>(&top_desc_, top_size.num(),
      top_size.channels(), top_size.height(), top_size.width(),
      top_stride.nstride(), top_stride.cstride(), top_stride.hstride(),
      top_stride.wstride());
  if (mode == "channel") {
    softmax_mode_ = CUDNN_SOFTMAX_MODE_CHANNEL;
  } else if (mode == "instance") {
    softmax_mode_ = CUDNN_SOFTMAX_MODE_INSTANCE;
  } else {
    LOG(FATAL) << "Unknown softmax mode " << mode;
  }
}

Softmax::~Softmax() {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
}

void Softmax::compute_cpu(const vector<bool>& add) {

}

void Softmax::compute_gpu(const vector<bool>& add) {
  DTYPE alpha = 1.;
  DTYPE beta = add[0] ? 1. : 0.;
  CUDNN_CHECK(cudnnSoftmaxForward(cudnn_handle(), CUDNN_SOFTMAX_ACCURATE,
          softmax_mode_, &alpha, bottom_desc_, inputs_[0]->gpu_data(), &beta,
          top_desc_, outputs_[0]->mutable_gpu_data()));
}

SoftmaxDown::SoftmaxDown(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
  std::tie(mode) = args;
  CHECK_EQ(inputs_[0]->size(), outputs_[0]->size());
  CHECK_EQ(inputs_[1]->size(), outputs_[0]->size());
  Size bottom_size = outputs_[0]->size();
  Size top_size = inputs_[0]->size();
  Stride bottom_stride = outputs_[0]->stride();
  Stride top_stride = inputs_[0]->stride();
  cudnn::createTensor4dDesc<DTYPE>(&bottom_desc_, bottom_size.num(),
      bottom_size.channels(), bottom_size.height(), bottom_size.width(),
      bottom_stride.nstride(), bottom_stride.cstride(),
      bottom_stride.hstride(), bottom_stride.wstride());
  cudnn::createTensor4dDesc<DTYPE>(&top_desc_, top_size.num(),
      top_size.channels(), top_size.height(), top_size.width(),
      top_stride.nstride(), top_stride.cstride(), top_stride.hstride(),
      top_stride.wstride());
  if (mode == "channel") {
    softmax_mode_ = CUDNN_SOFTMAX_MODE_CHANNEL;
  } else if (mode == "instance") {
    softmax_mode_ = CUDNN_SOFTMAX_MODE_INSTANCE;
  } else {
    LOG(FATAL) << "Unknown softmax mode " << mode;
  }
}

SoftmaxDown::~SoftmaxDown() {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
}

void SoftmaxDown::compute_cpu(const vector<bool>& add) {

}

void SoftmaxDown::compute_gpu(const vector<bool>& add) {
  DTYPE alpha = 1.;
  DTYPE beta = add[0] ? 1. : 0.;
  CUDNN_CHECK(cudnnSoftmaxBackward(cudnn_handle(), CUDNN_SOFTMAX_ACCURATE,
          softmax_mode_, &alpha, top_desc_, inputs_[1]->gpu_data(),
          top_desc_, inputs_[0]->gpu_data(), &beta, bottom_desc_,
          outputs_[0]->mutable_gpu_data()));
}

SoftmaxLoss::SoftmaxLoss(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
  CHECK_EQ(outputs_[0]->size(), Size(1, 1, 1, 1));
  CHECK_EQ(inputs_[1]->size().num(), inputs_[0]->size().num());
  CHECK_EQ(inputs_[1]->size().height(), inputs_[0]->size().height());
  CHECK_EQ(inputs_[1]->size().width(), inputs_[0]->size().width());
}

void SoftmaxLoss::compute_cpu(const vector<bool>& add) {
  const DTYPE* softmax_data = inputs_[0]->cpu_data();
  const DTYPE* label_data = inputs_[1]->cpu_data();
  Size softmax_size = inputs_[0]->size();
  int num = softmax_size.num();
  int dim = softmax_size.count() / num;
  int spatial_dim = softmax_size.height() * softmax_size.width();
  DTYPE loss = 0;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++) {
      loss -= log(std::max(softmax_data[i * dim + static_cast<int>(
          label_data[i * spatial_dim + j]) * spatial_dim + j], DTYPE(FLT_MIN)));
    }
  }
  *(outputs_[0]->mutable_cpu_data()) = loss / num / spatial_dim;
}

SoftmaxLossDown::SoftmaxLossDown(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs, const param_tuple& args)
    : Operation(inputs, outputs) {
  CHECK_EQ(inputs_[0]->size(), outputs_[0]->size());
}

void SoftmaxLossDown::compute_cpu(const vector<bool>& add) {
  DTYPE* bottom_diff = outputs_[0]->mutable_cpu_data();
  const DTYPE* softmax_data = inputs_[0]->cpu_data();
  Size softmax_size = inputs_[0]->size();
  caffe::caffe_cpu_copy(softmax_size.count(), softmax_data, bottom_diff);
  const DTYPE* label = inputs_[1]->cpu_data();
  int num = softmax_size.num();
  int dim = softmax_size.count() / num;
  int spatial_dim = softmax_size.height() * softmax_size.width();
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; ++j) {
      bottom_diff[i * dim + static_cast<int>(label[i * spatial_dim + j])
          * spatial_dim + j] -= 1;
    }
  }
  // Scale gradient
  const DTYPE loss_weight = inputs_[2]->cpu_data()[0];
  caffe::caffe_scal(softmax_size.count(), loss_weight / num / spatial_dim,
      bottom_diff);
}

}
