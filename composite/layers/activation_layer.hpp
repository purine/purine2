// Copyright Lin Min 2015
#ifndef PURINE_ACTIVATION_LAYER
#define PURINE_ACTIVATION_LAYER

#include "operations/include/activation.hpp"
#include "composite/layer.hpp"

namespace purine {

class ActivationLayer : public Layer {
 protected:
  string mode;
  bool inplace;
 public:
  typedef tuple<string, bool> param_tuple;
  ActivationLayer(const param_tuple& args, int rank, int device)
      : Layer(rank, device) {
    std::tie(mode, inplace) = args;
  }
  virtual ~ActivationLayer() {}

  virtual void setup() {
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
          create(bottom_size, "top"),
          create(bottom_size, "top_size")
        };
      } else {
        top_ = {
          create(bottom_[0]->shared_tensor(), "top"),
          create(bottom_[1]->shared_tensor(), "top_diff")
        };
      }
    }

    // create ops
    Op<Activation>* activation_up = create<Activation>(make_tuple(mode),
        "activation_up", "main");
    Op<ActivationDown>* activation_down = create<ActivationDown>(
        make_tuple(mode), "activation_down", "main");

    // forward
    B{ bottom_[0] } >> *activation_up >> B{ top_[0] };
    // backward
    B{ top_[1], top_[0] } >> *activation_down >> B{ bottom_[1] };
  }
};

}

#endif
