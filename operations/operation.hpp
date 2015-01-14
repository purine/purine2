// Copyright Lin Min 2015

#ifndef PURINE_OPERATION
#define PURINE_OPERATION

#include <tuple>
#include <vector>

#include "common/common.hpp"
#include "operations/tensor.hpp"
#include "operations/size.hpp"

using std::tuple;
using std::vector;

namespace purine {

class Operation {
 protected:
  vector<Tensor*> inputs_;
  vector<Tensor*> outputs_;
 public:
  explicit Operation(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs) : inputs_(inputs), outputs_(outputs) {
  }
  virtual ~Operation() {}
  virtual void compute_cpu(const vector<bool>& add) {
  }
  virtual void compute_gpu(const vector<bool>& add) {
  }
};

}

#endif  // PURINE_OPERATION
