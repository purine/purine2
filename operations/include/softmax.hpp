// Copyright Lin Min 2015
#ifndef PURINE_SOFTMAX
#define PURINE_SOFTMAX

#include "operations/operation.hpp"
#include "operations/cudnn.hpp"

namespace purine {

/**
 * { bottom } >> op >> { top }
 */
class Softmax : public Operation {
 protected:
  string mode;
  cudnnSoftmaxMode_t softmax_mode_;
  cudnnTensorDescriptor_t bottom_desc_ = NULL, top_desc_ = NULL;
 public:
  typedef tuple<string> param_tuple;
  explicit Softmax(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs, const param_tuple& args);
  virtual ~Softmax();
  void compute_cpu(const vector<bool>& add);
  void compute_gpu(const vector<bool>& add);
};

/**
 * { top_diff, top } >> op >> { bottom_diff }
 */
class SoftmaxDown : public Operation {
 protected:
  string mode;
  cudnnSoftmaxMode_t softmax_mode_;
  cudnnTensorDescriptor_t bottom_desc_ = NULL, top_desc_ = NULL;
 public:
  typedef tuple<string> param_tuple;
  explicit SoftmaxDown(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs, const param_tuple& args);
  virtual ~SoftmaxDown();
  void compute_cpu(const vector<bool>& add);
  void compute_gpu(const vector<bool>& add);
};

/**
 * { softmax, label } >> op >> { loss }
 */
class SoftmaxLoss : public Operation {
 public:
  typedef tuple<> param_tuple;
  explicit SoftmaxLoss(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs, const param_tuple& args);
  virtual void compute_cpu(const vector<bool>& add);
};

/**
 * { softmax, label, lambda } >> op >> { bottom_diff }
 */
class SoftmaxLossDown : public Operation {
 public:
  typedef tuple<> param_tuple;
  explicit SoftmaxLossDown(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs, const param_tuple& args);
  virtual void compute_cpu(const vector<bool>& add);
};

}

#endif
