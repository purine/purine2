// Copyright Lin Min 2015
#ifndef PURINE_ELTWISE
#define PURINE_ELTWISE

#include "operations/operation.hpp"

namespace purine {

/**
 * { ... } >> op >> { top }
 */
class Mul : public Operation {
 public:
  typedef tuple<> param_tuple;
  explicit Mul(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
      const param_tuple& args);
  virtual void compute_cpu(const vector<bool>& add);
  virtual void compute_gpu(const vector<bool>& add);
};

/**
 * { ... } >> op >> { top }
 */
class Sum : public Operation {
 public:
  typedef tuple<> param_tuple;
  explicit Sum(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
      const param_tuple& args);
  virtual void compute_cpu(const vector<bool>& add);
  virtual void compute_gpu(const vector<bool>& add);
};

/**
 * { ... } >> op >> { top }
 */
class WeightedSum : public Operation {
 protected:
  vector<DTYPE> weights_;
 public:
  typedef tuple<vector<DTYPE> > param_tuple;
  explicit WeightedSum(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs, const param_tuple& args);
  virtual void compute_cpu(const vector<bool>& add);
  virtual void compute_gpu(const vector<bool>& add);
  inline void set_weights(const vector<DTYPE>& w) { weights_ = w; }
  inline vector<DTYPE> weights() { return weights_; }
};

/**
 * { ... } >> op >> { top }
 */
class Average : public Operation {
 public:
  typedef tuple<> param_tuple;
  explicit Average(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs, const param_tuple& args);
  virtual void compute_cpu(const vector<bool>& add);
  virtual void compute_gpu(const vector<bool>& add);
};

class Scale : public Operation {
 protected:
  DTYPE scale;
 public:
  typedef tuple<DTYPE> param_tuple;
  explicit Scale(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
      const param_tuple& args);
  virtual void compute_cpu(const vector<bool>& add);
  virtual void compute_gpu(const vector<bool>& add);
};

}

#endif
