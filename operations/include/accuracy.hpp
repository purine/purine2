// Copyright Lin Min 2015
#ifndef PURINE_ACCURACY
#define PURINE_ACCURACY

#include "operations/operation.hpp"

namespace purine {

/**
 * { bottom, label } >> op >> { accuracy }
 */
class Accuracy : public Operation {
 protected:
  int topN;
 public:
  typedef tuple<int> param_tuple;
  explicit Accuracy(const vector<Tensor*>& inputs,
      const vector<Tensor*>& outputs, const param_tuple& args);
  virtual void compute_cpu(const vector<bool>& add);
};

}

#endif
