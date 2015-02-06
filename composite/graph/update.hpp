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
  DTYPE momentum;
  DTYPE learning_rate;
  DTYPE weight_decay;
 public:
  typedef tuple<DTYPE, DTYPE, DTYPE> param_tuple;
  Update(int rank, int device, const param_tuple& args)
      : Connectable(rank, device) {
    std::tie(momentum, learning_rate, weight_decay) = args;
  }
  virtual ~Update() override {}
 protected:
  virtual void setup() override;
};

}

#endif
