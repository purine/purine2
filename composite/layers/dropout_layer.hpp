// Copyright Lin Min 2015
#ifndef PURINE_DROPOUT_LAYER
#define PURINE_DROPOUT_LAYER

#include "composite/layer.hpp"
#include "operations/include/eltwise.hpp"
#include "operations/include/random.hpp"

namespace purine {

class DropoutLayer : public Layer {
 protected:
  DTYPE ratio;
  bool test;
  bool inplace;
 public:
  typedef tuple<DTYPE, bool, bool> param_tuple;
  DropoutLayer(int rank, int device, const param_tuple& args)
      : Layer(rank, device) {
    std::tie(ratio, test, inplace) = args;
  }
  virtual ~DropoutLayer() {}

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
          create("top", bottom_[1]->shared_tensor()),
          create("top_diff", bottom_[1]->shared_tensor())
        };
      }
    }
    if (!test) {
      // create ops
      Op<Bernoulli>* mask_generator = create<Bernoulli>("mask_gen", "main",
          make_tuple(1. - ratio)); // the thread can be other than main
      Blob* mask = create("mask", bottom_size);
      *mask_generator >> B{ mask };

      Op<Mul>* mul = create<Mul>("mul", "main", tuple<>());
      Op<Mul>* mul_down = create<Mul>("mul_down", "main", tuple<>());
      B{ bottom_[0], mask } >> *mul >> B{ top_[0] };
      B{ top_[1], mask } >> *mul_down >> B{ bottom_[1] };
    } else {
      Op<Scale>* scaler = create<Scale>("scaler", "main",
          make_tuple(1. - ratio));
      Op<Scale>* scaler_back = create<Scale>("scaler", "main",
          make_tuple(1. - ratio));
      B{ bottom_[0] } >> *scaler >> B{ top_[0] };
      B{ top_[1] } >> *scaler_back >> B{ bottom_[1] };
    }
  }
};

}

#endif
