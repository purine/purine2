// Copyright Lin Min 2015
#ifndef PURINE_LRN
#define PURINE_LRN

#include <vector>
#include <tuple>

#include "common/common.hpp"
#include "operations/tensor.hpp"
#include "operations/operation.hpp"

using std::tuple;
using std::vector;

namespace purine {

/**
 * { bottom, scale } >> op >> { top }
 */
class LRN : public Operation {
 protected:
  DTYPE alpha;
  DTYPE beta;
  int size;
 public:
  typedef tuple<DTYPE, DTYPE, int> param_tuple;
  explicit LRN(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
      const param_tuple& args);
  virtual void compute_gpu(const vector<bool>& add);
};

/**
 * { bottom } >> op >> { top }
 */
class LRNScale : public Operation {
 protected:
  DTYPE alpha;
  DTYPE beta;
  int size;
 public:
  typedef tuple<DTYPE, DTYPE, int> param_tuple;
  explicit LRNScale(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs, const param_tuple& args);
  virtual void compute_gpu(const vector<bool>& add);
};

/**
 * { bottom, top_diff, scale, top } >> op >> { bottom_diff }
 */
class LRNDown : public Operation {
 protected:
  DTYPE alpha;
  DTYPE beta;
  int size;
 public:
  typedef tuple<DTYPE, DTYPE, int> param_tuple;
  explicit LRNDown(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs, const param_tuple& args);
  virtual void compute_gpu(const vector<bool>& add);
};

}

#endif
