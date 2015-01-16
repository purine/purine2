// copyright Lin Min 2015

#ifndef PURINE_INNER
#define PURINE_INNER

#include "caffeine/caffeine.hpp"
#include "caffeine/math_functions.hpp"
#include "operations/operation.hpp"

namespace purine {

/**
 * { bottom, weight } >> op >> { top }
 */
class Inner : public Operation {
 public:
  typedef tuple<> param_tuple;
  explicit Inner(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
      const param_tuple& args);
  virtual void compute_gpu(const vector<bool>& add);
  virtual void compute_cpu(const vector<bool>& add);
};

/**
 * { top_diff, weight } >> op >> { bottom_diff }
 */
class InnerDown : public Operation {
 public:
  typedef tuple<> param_tuple;
  explicit InnerDown(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs, const param_tuple& args);
  virtual void compute_gpu(const vector<bool>& add);
  virtual void compute_cpu(const vector<bool>& add);
};

/**
 * { top_diff, bottom } >> op >> { weight_diff }
 */
class InnerWeight : public Operation {
 public:
  typedef tuple<> param_tuple;
  explicit InnerWeight(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs, const param_tuple& args);
  virtual void compute_gpu(const vector<bool>& add);
  virtual void compute_cpu(const vector<bool>& add);
};

}

#endif
