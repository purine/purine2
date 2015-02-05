// Copyright Lin Min 2015
#ifndef PURINE_ACTIVATION_LAYER
#define PURINE_ACTIVATION_LAYER

#include "operations/include/activation.hpp"
#include "composite/layer.hpp"

namespace purine {

typedef vector<Blob*> B;

class ActivationLayer : public Layer {
 protected:
  string mode;
  bool inplace;
 public:
  typedef tuple<string, bool> param_tuple;
  ActivationLayer(int rank, int device, const param_tuple& args)
      : Layer(rank, device) {
    std::tie(mode, inplace) = args;
  }
  virtual ~ActivationLayer() override {}

 protected:
  virtual void setup() override {
    CHECK(bottom_setup_);
    CHECK_EQ(bottom_.size(), 2);
    Size bottom_size = bottom_[0]->tensor()->size();

    // check top
    if (top_.size() != 0) {
      CHECK_EQ(top_.size(), 2);
      for (auto top : top_) {
        CHECK_EQ(top->tensor()->size(), bottom_size);
      }
    } else {
      if (!inplace) {
        top_ = {
          create("top", bottom_size),
          create("top_diff", bottom_size)
        };
      } else {
        top_ = {
          create("top", bottom_[0]->shared_tensor()),
          create("top_diff", bottom_[1]->shared_tensor())
        };
      }
    }

    // create ops
    Op<Activation>* activation_up = create<Activation>("activation_up", "main",
        make_tuple(mode));
    Op<ActivationDown>* activation_down = create<ActivationDown>(
        "activation_down", "main", make_tuple(mode));

    // forward
    B{ bottom_[0] } >> *activation_up >> B{ top_[0] };
    // backward
    B{ top_[1], top_[0] } >> *activation_down >> B{ bottom_[1] };
  }
};

}

#endif
