// Copyright Lin Min 2015
#ifndef PURINE_COPY
#define PURINE_COPY

#include "operations/operation.hpp"

namespace purine {

class MemCopy : public Operation {
 public:
  typedef tuple<> param_tuple;
  explicit MemCopy(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
      const param_tuple& args);
  virtual void compute_cpu(const vector<bool>& add);
  virtual void compute_gpu(const vector<bool>& add);
};

}

#endif
