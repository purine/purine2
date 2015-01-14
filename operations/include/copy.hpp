// Copyright Lin Min 2015
#ifndef PURINE_COPY
#define PURINE_COPY

#include "operations/operation.hpp"

namespace purine {

class Copy : public Operation {
 public:
  explicit Copy(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs);
  virtual void compute_cpu(const vector<bool>& add);
  virtual void compute_gpu(const vector<bool>& add);
};

}

#endif
