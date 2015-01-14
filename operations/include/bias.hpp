// Copyright Lin Min 2015
#ifndef PURINE_BIAS
#define PURINE_BIAS

#include "operations/operation.hpp"
#include "caffeine/cudnn.hpp"

namespace purine {

/**
 * { bias } >> op >> { top }
 */
class Bias : public Operation {
 protected:
  cudnnTensorDescriptor_t bias_desc_ = NULL, top_desc_ = NULL;
 public:
  explicit Bias(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs);
  virtual ~Bias();
  virtual void compute_cpu(const vector<bool>& add);
  virtual void compute_gpu(const vector<bool>& add);
};

/**
 * { top_diff } >> op >> { bias_diff }
 */
class BiasDown : public Operation {
 protected:
  cudnnTensorDescriptor_t bias_desc_ = NULL, top_desc_ = NULL;
 public:
  explicit BiasDown(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs);
  virtual ~BiasDown();
  virtual void compute_cpu(const vector<bool>& add);
  virtual void compute_gpu(const vector<bool>& add);
};

}

#endif
