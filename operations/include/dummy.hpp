// Copyright Lin Min 2015
#ifndef PURINE_DUMMY
#define PURINE_DUMMY

#include "operations/operation.hpp"

namespace purine {

class Dummy : public Operation {
 public:
  typedef tuple<> param_tuple;
  explicit Dummy(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs, const param_tuple& args);
  virtual void compute_cpu(const vector<bool>& add);
  virtual void compute_gpu(const vector<bool>& add);
};

}

#endif
