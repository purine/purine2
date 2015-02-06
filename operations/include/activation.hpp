// Copyright Lin Min 2015
#ifndef PURINE_ACTIVATION
#define PURINE_ACTIVATION

#include <string>
#include "caffeine/cudnn.hpp"
#include "operations/operation.hpp"

using std::string;

namespace purine {

/**
 * { bottom } >> op >> { top }
 */
class Activation : public Operation {
 protected:
  string mode_;
  cudnnTensorDescriptor_t bottom_desc_, top_desc_;
  cudnnActivationMode_t activation_mode_;
 public:
  typedef tuple<string> param_tuple;
  explicit Activation(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs, const param_tuple& args);
  virtual ~Activation();
  virtual void compute_gpu(const vector<bool>& add);
};

/**
 * { top_diff, top } >> op >> { bottom_diff }
 */
class ActivationDown : public Operation {
 protected:
  string mode_;
  cudnnTensorDescriptor_t bottom_desc_, top_desc_;
  cudnnActivationMode_t activation_mode_;
 public:
  typedef tuple<string> param_tuple;
  explicit ActivationDown(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs, const param_tuple& args);
  virtual ~ActivationDown();
  virtual void compute_gpu(const vector<bool>& add);
};

}

#endif
