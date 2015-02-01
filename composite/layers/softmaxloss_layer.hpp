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
      create({1, 1, 1, 1}, "loss", rank_, -1),
      create({1, 1, 1, 1}, "loss_diff", rank_, -1)
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
    Op<Softmax>* softmax = create<Softmax>(make_tuple("channel"),
        "softmax", "main");
    // create on CPU
    Op<SoftmaxLoss>* softmaxloss = create<SoftmaxLoss>(
        SoftmaxLoss::param_tuple(), "softmaxloss", rank_, -1, "main");
    Op<SoftmaxDown>* softmaxloss_down = create<SoftmaxLossDown>(
        SoftmaxLossDown::param_tuple(), "softmaxloss_down", rank_, -1, "main");

    // forward
    Blob* label_cpu =
        (B{ bottom_[2] } >> *create<Copy>("...", rank_, -1)).top()[0];

    Blob* softmaxed_cpu = (B{ bottom_[0] } >> *softmax >>
        B{ create(bottom_[0]->tensor()->size(), "softmaxed") } >>
        *create<Copy>("...", rank_, -1)).top()[0];

    B{ softmaxed_cpu, label_cpu } >> *softmaxloss >> B{ loss_[0] };
    // backward
    B{ softmaxed_cpu, label_cpu, loss_[1] } >>
        *softmaxloss_down >> B{ create(bottom_[1]->tensor()->size(),
              "...", rank_, -1) } >> *create<Copy>("...") >> B{ bottom_[1] };
  }
};

}

#endif
