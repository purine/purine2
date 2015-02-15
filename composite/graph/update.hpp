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
  void set_momentum(DTYPE momentum) {
    vector<DTYPE> weights = compute_update->operation()->weights();
    weights[0] = momentum;
    compute_update->operation()->set_weights(weights);
  }
  void set_learning_rate(DTYPE learning_rate) {
    vector<DTYPE> weights = compute_update->operation()->weights();
    weights[1] = learning_rate;
    compute_update->operation()->set_weights(weights);
  }
  void set_weight_decay(DTYPE weight_decay) {
    vector<DTYPE> weights = compute_update->operation()->weights();
    weights[2] = weight_decay;
    compute_update->operation()->set_weights(weights);
  }
 protected:
  virtual void setup() override;
  Op<WeightedSum>* compute_update = NULL;
};

}

#endif
