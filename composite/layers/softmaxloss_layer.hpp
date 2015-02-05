// Copyright Lin Min 2015
#ifndef PURINE_SOFTMAXLOSS_LAYER
#define PURINE_SOFTMAXLOSS_LAYER

#include "composite/layer.hpp"
#include "operations/include/softmax.hpp"

namespace purine {

class SoftmaxLossLayer : public Layer {
 protected:
  DTYPE loss_scale = 1.;
 public:
  typedef tuple<DTYPE> param_tuple;
  SoftmaxLossLayer(int rank, int device, const param_tuple& args)
      : Layer(rank, device) {
    std::tie(loss_scale) = args;
    loss_ = {
      create("loss", rank_, -1, {1, 1, 1, 1}),
      create("loss_diff", rank_, -1, {1, 1, 1, 1})
    };
    // set loss_scale
    loss_[1]->mutable_cpu_data()[0] = -1 * loss_scale;
    top_setup_ = true;
  }
  virtual ~SoftmaxLossLayer() override {}
 protected:
  virtual void setup() {
    CHECK(bottom_setup_);
    CHECK_EQ(bottom_.size(), 3);
    Size bottom_size = bottom_[0]->tensor()->size();

    // check top, softmaxloss has no top.
    CHECK_EQ(top_.size(), 0);

    // create ops
    Op<Softmax>* softmax = create<Softmax>("softmax", "main",
        make_tuple("channel"));
    // create on CPU
    Op<SoftmaxLoss>* softmaxloss = create<SoftmaxLoss>("softmaxloss", rank_,
        -1, "main", SoftmaxLoss::param_tuple());
    Op<SoftmaxDown>* softmaxloss_down = create<SoftmaxLossDown>(
        "softmaxloss_down", rank_, -1, "main", SoftmaxLossDown::param_tuple());

    // forward
    Blob* label_cpu =
        (B{ bottom_[2] } >> *create<Copy>("...",
            Copy::param_tuple(rank_, -1))).top()[0];

    Blob* softmaxed_cpu = (B{ bottom_[0] } >> *softmax >>
        B{ create("softmaxed", bottom_[0]->tensor()->size()) } >>
        *create<Copy>("...", Copy::param_tuple(rank_, -1))).top()[0];

    B{ softmaxed_cpu, label_cpu } >> *softmaxloss >> B{ loss_[0] };
    // backward
    B{ softmaxed_cpu, label_cpu, loss_[1] } >>
        *softmaxloss_down >> B{ create("...", rank_, -1,
              bottom_[1]->tensor()->size()) }
        >> *create<Copy>("...", Copy::param_tuple()) >> B{ bottom_[1] };
  }
};

}

#endif
