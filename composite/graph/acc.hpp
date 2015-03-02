// Copyright Lin Min 2015
#ifndef PURINE_ACC_LAYER
#define PURINE_ACC_LAYER

#include "composite/layer.hpp"
#include "operations/include/accuracy.hpp"

namespace purine {

class Acc : public Connectable {
 protected:
  Blob* label_;
  int top_N_;
  vector<Blob*> loss_;
 public:
  typedef tuple<int> param_tuple;
  Acc(int rank, int device, const param_tuple& args)
      : Connectable(rank, device) {
    std::tie(top_N_) = args;
    loss_ = {
      create("loss", rank_, -1, {1, 1, 1, 1}),
    };
    top_setup_ = true;
  }
  virtual ~Acc() override {}
  void set_label(Blob* label) { label_ = label; }
  const vector<Blob*>& loss() { return loss_; }
 protected:
  virtual void setup() {
    CHECK(bottom_setup_);
    CHECK_EQ(bottom_.size(), 1);
    Size bottom_size = bottom_[0]->tensor()->size();

    // check top, softmaxloss has no top.
    CHECK_EQ(top_.size(), 0);

    // forward
    Blob* label_cpu = (B{ label_ } >> *createAny<Copy>("label_cpu",
            Copy::param_tuple(rank_, -1))).top()[0];
    Blob* output_cpu = (bottom_ >> *createAny<Copy>("output_cpu",
            Copy::param_tuple(rank_, -1))).top()[0];

    B{ output_cpu, label_cpu } >>
        *create<Accuracy>("acc", "", Accuracy::param_tuple(top_N_)) >> loss_;
  }
};

}

#endif
