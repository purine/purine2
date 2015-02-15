// Copyright Lin Min 2015
#ifndef PURINE_PARAM_SERVER
#define PURINE_PARAM_SERVER

#include "dispatch/graph_template.hpp"
#include "composite/connectable.hpp"
#include "composite/graph/update.hpp"

namespace purine {

/**
 * weight_diffs are from different minions
 * new_weights are delivered to different minions
 * { weight_diffs } >> param_server >> { new_weights }
 */
class AllReduce : public Connectable {
 protected:
  // current weight kept by param server
  Blob* weight_;
  Blob* weight_diff_;
  Blob* history_;
  Update::param_tuple args_;
 public:
  typedef Update::param_tuple param_tuple;
  explicit AllReduce(int rank, int device, const param_tuple& args)
      : Connectable(rank, device), args_(args) {
  }
  virtual ~AllReduce() override {}
  shared_ptr<Tensor> weight() { return weight_->shared_tensor(); }
  shared_ptr<Tensor> weight_diff() { return weight_diff_->shared_tensor(); }
  shared_ptr<Tensor> history() { return history_->shared_tensor(); }
 protected:
  void setup() {
    CHECK(bottom_setup_);
    int bottom_num = bottom_.size();
    CHECK_GT(bottom_.size(), 0);
    Size bottom_size = bottom_[0]->tensor()->size();
    // create top
    top_ = vector<Blob*>(bottom_num);
    for (int i = 0; i < bottom_num; ++i) {
      top_[i] = create("new_weight_" + to_string(i), bottom_[i]->rank(),
          bottom_[i]->device(), bottom_[i]->tensor()->size());
    }
    // agg bottom
    weight_diff_ = create("[weight_diff]", bottom_size);
    Aggregate* agg = createAny<Aggregate>("agg_diff",
        Aggregate::param_tuple(Aggregate::AVERAGE, rank_, device_));
    bottom_ >> *agg >> vector<Blob*>{ weight_diff_ };
    // create history, weight
    weight_ = create("[weight]", bottom_size);
    history_ = create("[history]", bottom_size);
    Blob* new_weight = create("[new_weight]", weight_->shared_tensor());
    Blob* new_history = create("[new_history]", history_->shared_tensor());
    Update* updator = createGraph<Update>("updator", args_);
    vector<Blob*>{ weight_, weight_diff_, history_ } >>
    *updator >> vector<Blob*>{ new_weight, new_history };
    // distribute
    vector<Blob*>{ new_weight } >> *createAny<Distribute>("dist_new_weight",
        Distribute::param_tuple()) >> top_;
    // intialize history
    Runnable fill_history(rank_, device_);
    Blob* to_fill = fill_history.create("history", history_->shared_tensor());
    *fill_history.create<Constant>("filler", "", Constant::param_tuple(0.))
        >> vector<Blob*>{ to_fill };
    fill_history.run();
  }
};

}

#endif
