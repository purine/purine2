// Copyright Lin Min 2015
#ifndef PURINE_UPDATE
#define PURINE_UPDATE

#include <utility>
#include "dispatch/graph_template.hpp"
#include "composite/connectable.hpp"
#include "operations/include/eltwise.hpp"

namespace purine {

/**
 * { weight, weight_diff, history } >> update >> { new_weight, new_history }
 */
class Update : public Connectable {
 protected:
  DTYPE momentum_;
  DTYPE learning_rate_;
  DTYPE weight_decay_;
 public:
  typedef tuple<DTYPE, DTYPE, DTYPE> param_tuple;
  Update(int rank, int device, const param_tuple& args)
      : Connectable(rank, device) {
    std::tie(momentum_, learning_rate_, weight_decay_) = args;
  }
  virtual ~Update() override {}
  // set parameters.
  inline void set_param(const WeightedSum::param_tuple& param) {
    compute_update->set_param(param);
  }
 protected:
  virtual void setup() override;
  Op<WeightedSum>* compute_update = NULL;
};

}

#endif
