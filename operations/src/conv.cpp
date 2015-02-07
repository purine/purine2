// Copyright Lin Min 2015
#include "operations/include/conv.hpp"

namespace purine {

// Update cudnn R2
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
  cudnn::createTensor4dDesc<DTYPE>(&bottom_desc_, bottom_size.num(),
      bottom_size.channels(), bottom_size.height(), bottom_size.width(),
      bottom_stride.nstride(), bottom_stride.cstride(),
      bottom_stride.hstride(), bottom_stride.wstride());
  cudnn::createTensor4dDesc<DTYPE>(&top_desc_, top_size.num(),
      top_size.channels(), top_size.height(), top_size.width(),
      top_stride.nstride(), top_stride.cstride(), top_stride.hstride(),
      top_stride.wstride());
  cudnn::createFilterDesc<DTYPE>(&filter_desc_, kernel_size.num(),
      kernel_size.channels(), kernel_size.height(), kernel_size.width());
  cudnn::createConvolutionDesc<DTYPE>(&conv_desc_, pad_h, pad_w, stride_h,
      stride_w);
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(cudnn_handle(), bottom_desc_,
          filter_desc_, conv_desc_, top_desc_,
          CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo_));
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
          bottom_desc_, filter_desc_, conv_desc_, top_desc_, algo_,
          &workspace_size_));
}

Conv::~Conv() {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
  CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
  CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));
}

void Conv::compute_gpu(const vector<bool>& add) {
  if (!workspace_ && workspace_size_ != 0) {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    workspace_.reset(new Tensor(current_rank(), device,
            {1, 1, 1, workspace_size_}));
  }
  DTYPE alpha = 1.;
  DTYPE beta = add[0] ? 1. : 0.;
  CUDNN_CHECK(cudnnConvolutionForward(cudnn_handle(), &alpha, bottom_desc_,
          inputs_[0]->gpu_data(), filter_desc_, inputs_[1]->gpu_data(),
          conv_desc_, algo_, workspace_ ? workspace_->mutable_gpu_data() : 0,
          workspace_size_, &beta, top_desc_, outputs_[0]->mutable_gpu_data()));
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
  cudnn::createTensor4dDesc<DTYPE>(&bottom_desc_, bottom_size.num(),
      bottom_size.channels(), bottom_size.height(), bottom_size.width(),
      bottom_stride.nstride(), bottom_stride.cstride(),
      bottom_stride.hstride(), bottom_stride.wstride());
  cudnn::createTensor4dDesc<DTYPE>(&top_desc_, top_size.num(),
      top_size.channels(), top_size.height(), top_size.width(),
      top_stride.nstride(), top_stride.cstride(), top_stride.hstride(),
      top_stride.wstride());
  cudnn::createFilterDesc<DTYPE>(&filter_desc_, kernel_size.num(),
      kernel_size.channels(), kernel_size.height(), kernel_size.width());
  cudnn::createConvolutionDesc<DTYPE>(&conv_desc_, pad_h, pad_w, stride_h,
      stride_w);
}

ConvDown::~ConvDown() {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
  CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
  CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));
}

void ConvDown::compute_gpu(const vector<bool>& add) {
  const DTYPE* weight_data = inputs_[1]->gpu_data();
  const DTYPE* top_diff = inputs_[0]->gpu_data();
  DTYPE* bottom_diff = outputs_[0]->mutable_gpu_data();
  DTYPE alpha = 1.;
  DTYPE beta = add[0] ? 1. : 0.;
  CUDNN_CHECK(cudnnConvolutionBackwardData(cudnn_handle(), &alpha,
          filter_desc_, weight_data, top_desc_, top_diff, conv_desc_, &beta,
          bottom_desc_, bottom_diff));
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
  cudnn::createTensor4dDesc<DTYPE>(&bottom_desc_, bottom_size.num(),
      bottom_size.channels(), bottom_size.height(), bottom_size.width(),
      bottom_stride.nstride(), bottom_stride.cstride(),
      bottom_stride.hstride(), bottom_stride.wstride());
  cudnn::createTensor4dDesc<DTYPE>(&top_desc_, top_size.num(),
      top_size.channels(), top_size.height(), top_size.width(),
      top_stride.nstride(), top_stride.cstride(), top_stride.hstride(),
      top_stride.wstride());
  cudnn::createFilterDesc<DTYPE>(&filter_desc_, kernel_size.num(),
      kernel_size.channels(), kernel_size.height(), kernel_size.width());
  cudnn::createConvolutionDesc<DTYPE>(&conv_desc_, pad_h, pad_w, stride_h,
      stride_w);
}

ConvWeight::~ConvWeight() {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
  CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
  CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));
}

void ConvWeight::compute_gpu(const vector<bool>& add) {
  const DTYPE* top_diff = inputs_[0]->gpu_data();
  const DTYPE* bottom_data = inputs_[1]->gpu_data();
  DTYPE* weight_diff = outputs_[0]->mutable_gpu_data();
  DTYPE alpha = 1.;
  DTYPE beta = add[0] ? 1. : 0.;
  CUDNN_CHECK(cudnnConvolutionBackwardFilter(cudnn_handle(), &alpha,
          bottom_desc_, bottom_data, top_desc_, top_diff, conv_desc_, &beta,
          filter_desc_, weight_diff));
}

}
