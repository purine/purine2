// Copyright Lin Min 2015
#ifndef PURINE_RANDOM
#define PURINE_RANDOM

#include "operations/tensor.hpp"
#include "operations/operation.hpp"

namespace purine {

/**
 * {} >> op >> { top, ... }
 */
class Gaussian : public Operation {
 protected:
  DTYPE mean;
  DTYPE std;
 public:
  typedef tuple<DTYPE, DTYPE> param_tuple;
  explicit Gaussian(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs, const param_tuple& args);
  virtual void compute_cpu(const vector<bool>& add);
  virtual void compute_gpu(const vector<bool>& add);
};

/**
 * {} >> op >> { top, ... }
 */
class Uniform : public Operation {
 protected:
  DTYPE min;
  DTYPE max;
 public:
  typedef tuple<DTYPE, DTYPE> param_tuple;
  explicit Uniform(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs, const param_tuple& args);
  virtual void compute_cpu(const vector<bool>& add);
  virtual void compute_gpu(const vector<bool>& add);
};

/**
 * {} >> op >> { top, ... }
 */
class Constant : public Operation {
 protected:
  DTYPE constant;
 public:
  typedef tuple<DTYPE> param_tuple;
  explicit Constant(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs, const param_tuple& args);
  virtual void compute_cpu(const vector<bool>& add);
  virtual void compute_gpu(const vector<bool>& add);
};

/**
 * {} >> op >> { top, ... }
 */
class Bernoulli : public Operation {
 protected:
  DTYPE prob;
 public:
  typedef tuple<DTYPE> param_tuple;
  explicit Bernoulli(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs, const param_tuple& args);
  virtual void compute_cpu(const vector<bool>& add);
  virtual void compute_gpu(const vector<bool>& add);
};

}

#endif
