// Copyright Lin Min 2015

#ifndef PURINE_CONV
#define PURINE_CONV

#include "operations/operation.hpp"
#include "operations/cudnn.hpp"
#include "operations/tensor.hpp"

namespace purine {

/**
 * { bottom, weight } >> op >> { top }
 */
class Conv : public Operation {
 protected:
  int pad_h, pad_w, stride_h, stride_w;
  cudnnTensorDescriptor_t bottom_desc_ = NULL;
  cudnnTensorDescriptor_t top_desc_ = NULL;
  cudnnFilterDescriptor_t filter_desc_ = NULL;
  cudnnConvolutionDescriptor_t conv_desc_ = NULL;
  cudnnConvolutionFwdAlgo_t algo_ = (cudnnConvolutionFwdAlgo_t)NULL;
  size_t workspace_size_ = 0;
  shared_ptr<Tensor> workspace_;
 public:
  typedef tuple<int, int, int, int> param_tuple;
  explicit Conv(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
      const param_tuple& args);
  virtual ~Conv();
  virtual void compute_gpu(const vector<bool>& add);
};

/**
 * { top_diff, weight } >> op >> { bottom_diff }
 */
class ConvDown : public Operation {
 protected:
  int pad_h, pad_w, stride_h, stride_w;
  cudnnTensorDescriptor_t bottom_desc_ = NULL;
  cudnnTensorDescriptor_t top_desc_ = NULL;
  cudnnFilterDescriptor_t filter_desc_ = NULL;
  cudnnConvolutionDescriptor_t conv_desc_ = NULL;
 public:
  typedef tuple<int, int, int, int> param_tuple;
  explicit ConvDown(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs,
      const param_tuple& args);
  virtual ~ConvDown();
  virtual void compute_gpu(const vector<bool>& add);
};

/**
 * { top_diff, bottom } >> op >> { weight_diff }
 */
class ConvWeight : public Operation {
 protected:
  int pad_h, pad_w, stride_h, stride_w;
  cudnnTensorDescriptor_t bottom_desc_ = NULL;
  cudnnTensorDescriptor_t top_desc_ = NULL;
  cudnnFilterDescriptor_t filter_desc_ = NULL;
  cudnnConvolutionDescriptor_t conv_desc_ = NULL;
 public:
  typedef tuple<int, int, int, int> param_tuple;
  explicit ConvWeight(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs,
      const param_tuple& args);
  virtual ~ConvWeight();
  virtual void compute_gpu(const vector<bool>& add);
};

}

#endif
