// Copyright Lin Min 2015
#ifndef PURINE_POOL
#define PURINE_POOL

#include "caffeine/cudnn.hpp"
#include "operations/operation.hpp"

namespace purine {

/**
 * { bottom } >> op >> { top }
 */
class Pool : public Operation {
 protected:
  string method;
  int kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w;
  cudnnTensorDescriptor_t bottom_desc_ = NULL;
  cudnnTensorDescriptor_t top_desc_ = NULL;
  cudnnPoolingDescriptor_t pool_desc_ = NULL;
 public:
  typedef tuple<string, int, int, int, int, int, int> param_tuple;
  explicit Pool(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
      const param_tuple& args);
  virtual ~Pool();
  virtual void compute_gpu(const vector<bool>& add);
  virtual void compute_cpu(const vector<bool>& add);
};

/**
 * { top_diff, top, bottom } >> op >> { bottom_diff }
 */
class PoolDown : public Operation {
 protected:
  string method;
  int kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w;
  cudnnTensorDescriptor_t bottom_desc_ = NULL;
  cudnnTensorDescriptor_t top_desc_ = NULL;
  cudnnPoolingDescriptor_t pool_desc_ = NULL;
 public:
  typedef tuple<string, int, int, int, int, int, int> param_tuple;
  explicit PoolDown(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs, const param_tuple& args);
  virtual ~PoolDown();
  virtual void compute_gpu(const vector<bool>& add);
  virtual void compute_cpu(const vector<bool>& add);
};

}

#endif
