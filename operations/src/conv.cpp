// Copyright Lin Min 2015
#include "operations/include/conv.hpp"

namespace purine {

Conv::Conv(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
    const param_tuple& args) : Operation(inputs, outputs) {
  std::tie(pad_h, pad_w, stride_h, stride_w) = args;
  CHECK_EQ(inputs_.size(), 2);
  CHECK_EQ(outputs_.size(), 1);
  Size bottom_size = inputs_[0]->size();
  Stride bottom_stride = inputs_[0]->stride();
  Size top_size = outputs_[0]->size();
  Stride top_stride = outputs_[0]->stride();
  Size kernel_size = inputs_[1]->size();
  CHECK_EQ(bottom_size.num(), top_size.num());
  CHECK_EQ(bottom_size.channels(), kernel_size.channels());
  CHECK_EQ(kernel_size.num(), top_size.channels());
  CHECK_EQ((bottom_size.height() + 2 * pad_h - kernel_size.height())
      / stride_h + 1, top_size.height());
  CHECK_EQ((bottom_size.width() + 2 * pad_w - kernel_size.width())
      / stride_w + 1, top_size.width());
  cudnn::createTensor4dDescV1<DTYPE>(&bottom_desc_, bottom_size, bottom_stride);
  cudnn::createTensor4dDescV1<DTYPE>(&top_desc_, top_size, top_stride);
  cudnn::createFilterDescV1<DTYPE>(&filter_desc_, kernel_size);
  cudnn::createConvolutionDescV1<DTYPE>(&conv_desc_, bottom_desc_, filter_desc_,
      pad_h, pad_w, stride_h, stride_w);
}

Conv::~Conv() {
  CUDNN_CHECK(cudnn_v1::cudnnDestroyTensor4dDescriptor(bottom_desc_));
  CUDNN_CHECK(cudnn_v1::cudnnDestroyTensor4dDescriptor(top_desc_));
  CUDNN_CHECK(cudnn_v1::cudnnDestroyFilterDescriptor(filter_desc_));
  CUDNN_CHECK(cudnn_v1::cudnnDestroyConvolutionDescriptor(conv_desc_));
}

void Conv::compute_gpu(const vector<bool>& add) {
  CUDNN_CHECK(cudnn_v1::cudnnConvolutionForward(cudnnv1_handle(), bottom_desc_,
          inputs_[0]->gpu_data(), filter_desc_, inputs_[1]->gpu_data(),
          conv_desc_, top_desc_, outputs_[0]->mutable_gpu_data(), add[0] ?
          cudnn_v1::CUDNN_RESULT_ACCUMULATE :
          cudnn_v1::CUDNN_RESULT_NO_ACCUMULATE));
}

ConvDown::ConvDown(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs,
    const param_tuple& args) : Operation(inputs, outputs) {
  std::tie(pad_h, pad_w, stride_h, stride_w) = args;
  Size bottom_size = outputs_[0]->size();
  Stride bottom_stride = outputs_[0]->stride();
  Size top_size = inputs_[0]->size();
  Stride top_stride = inputs_[0]->stride();
  Size kernel_size = inputs_[1]->size();
  CHECK_EQ(bottom_size.num(), top_size.num());
  CHECK_EQ(bottom_size.channels(), kernel_size.channels());
  CHECK_EQ(kernel_size.num(), top_size.channels());
  CHECK_EQ((bottom_size.height() + 2 * pad_h - kernel_size.height())
      / stride_h + 1, top_size.height());
  CHECK_EQ((bottom_size.width() + 2 * pad_w - kernel_size.width())
      / stride_w + 1, top_size.width());
  cudnn::createTensor4dDescV1<DTYPE>(&bottom_desc_, bottom_size, bottom_stride);
  cudnn::createTensor4dDescV1<DTYPE>(&top_desc_, top_size, top_stride);
  cudnn::createFilterDescV1<DTYPE>(&filter_desc_, kernel_size);
  cudnn::createConvolutionDescV1<DTYPE>(&conv_desc_, bottom_desc_, filter_desc_,
      pad_h, pad_w, stride_h, stride_w);
}

ConvDown::~ConvDown() {
  CUDNN_CHECK(cudnn_v1::cudnnDestroyTensor4dDescriptor(bottom_desc_));
  CUDNN_CHECK(cudnn_v1::cudnnDestroyTensor4dDescriptor(top_desc_));
  CUDNN_CHECK(cudnn_v1::cudnnDestroyFilterDescriptor(filter_desc_));
  CUDNN_CHECK(cudnn_v1::cudnnDestroyConvolutionDescriptor(conv_desc_));
}

void ConvDown::compute_gpu(const vector<bool>& add) {
  const DTYPE* weight_data = inputs_[1]->gpu_data();
  const DTYPE* top_diff = inputs_[0]->gpu_data();
  DTYPE* bottom_diff = outputs_[0]->mutable_gpu_data();
  CUDNN_CHECK(cudnn_v1::cudnnConvolutionBackwardData(cudnnv1_handle(), filter_desc_,
          weight_data, top_desc_, top_diff, conv_desc_, bottom_desc_,
          bottom_diff, add[0] ? cudnn_v1::CUDNN_RESULT_ACCUMULATE :
          cudnn_v1::CUDNN_RESULT_NO_ACCUMULATE));
}

ConvWeight::ConvWeight(const vector<Tensor*>& inputs,
    const vector<Tensor*>& outputs,
    const param_tuple& args) : Operation(inputs, outputs) {
  std::tie(pad_h, pad_w, stride_h, stride_w) = args;
  Size bottom_size = inputs_[1]->size();
  Size top_size = inputs_[0]->size();
  Size kernel_size = outputs_[0]->size();
  Stride bottom_stride = inputs_[1]->stride();
  Stride top_stride = inputs_[0]->stride();
  CHECK_EQ(bottom_size.num(), top_size.num());
  CHECK_EQ(bottom_size.channels(), kernel_size.channels());
  CHECK_EQ(kernel_size.num(), top_size.channels());
  CHECK_EQ((bottom_size.height() + 2 * pad_h - kernel_size.height())
      / stride_h + 1, top_size.height());
  CHECK_EQ((bottom_size.width() + 2 * pad_w - kernel_size.width())
      / stride_w + 1, top_size.width());
  cudnn::createTensor4dDescV1<DTYPE>(&bottom_desc_, bottom_size, bottom_stride);
  cudnn::createTensor4dDescV1<DTYPE>(&top_desc_, top_size, top_stride);
  cudnn::createFilterDescV1<DTYPE>(&filter_desc_, kernel_size);
  cudnn::createConvolutionDescV1<DTYPE>(&conv_desc_, bottom_desc_, filter_desc_,
      pad_h, pad_w, stride_h, stride_w);
}

ConvWeight::~ConvWeight() {
  CUDNN_CHECK(cudnn_v1::cudnnDestroyTensor4dDescriptor(bottom_desc_));
  CUDNN_CHECK(cudnn_v1::cudnnDestroyTensor4dDescriptor(top_desc_));
  CUDNN_CHECK(cudnn_v1::cudnnDestroyFilterDescriptor(filter_desc_));
  CUDNN_CHECK(cudnn_v1::cudnnDestroyConvolutionDescriptor(conv_desc_));
}

void ConvWeight::compute_gpu(const vector<bool>& add) {
  const DTYPE* top_diff = inputs_[0]->gpu_data();
  const DTYPE* bottom_data = inputs_[1]->gpu_data();
  DTYPE* weight_diff = outputs_[0]->mutable_gpu_data();
  CUDNN_CHECK(cudnn_v1::cudnnConvolutionBackwardFilter(cudnnv1_handle(),
          bottom_desc_, bottom_data, top_desc_, top_diff, conv_desc_,
          filter_desc_, weight_diff, add[0] ? cudnn_v1::CUDNN_RESULT_ACCUMULATE
          : cudnn_v1::CUDNN_RESULT_NO_ACCUMULATE));
}

}
