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
  explicit Inner(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs);
  virtual ~Inner();
  virtual void compute_gpu(const vector<bool>& add);
  virtual void compute_cpu(const vector<bool>& add);
};

/**
 * { top_diff, weight } >> op >> { bottom_diff }
 */
class InnerDown : public Operation {
 public:
  explicit InnerDown(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs);
  virtual ~InnerDown();
  virtual void compute_gpu(const vector<bool>& add);
  virtual void compute_cpu(const vector<bool>& add);
};

/**
 * { top_diff, bottom } >> op >> { weight_diff }
 */
class InnerWeight : public Operation {
 public:
  explicit InnerWeight(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs);
  virtual ~InnerWeight();
  virtual void compute_gpu(const vector<bool>& add);
  virtual void compute_cpu(const vector<bool>& add);
};

}

#endif
