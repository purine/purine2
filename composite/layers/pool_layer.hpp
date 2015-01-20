// Copyright Lin Min 2015
#ifndef PURINE_POOL_LAYER
#define PURINE_POOL_LAYER

#include "composite/layer.hpp"
#include "operations/include/pool.hpp"

namespace purine {

class PoolLayer : public Layer {
 protected:
  Pool::param_tuple args_;
 public:
  typedef Pool::param_tuple param_tuple;
  PoolLayer(const param_tuple& args, int rank, int device)
      : Layer(rank, device), args_(args) {
  }
  virtual void ~PoolLayer() override {}

 protected:
  virtual void setup() override {
    CHECK(bottom_setup_);
    CHECK_EQ(bottom_.size(), 2);
    Size bottom_size = bottom_[0]->tensor()->size();
    int out_h = static_cast<int>(ceil(static_cast<float>(bottom_size.height()
                - kernel_h) / stride_h)) + 1;
    int out_w = static_cast<int>(ceil(static_cast<float>(bottom_size.width()
                - kernel_w) / stride_w)) + 1;
    Size expect_top_size = { bottom_size.num(), bottom_size.channels(),
                             out_h, out_w };

    // check top
    if (top_.size() != 0) {
      CHECK_EQ(top_.size(), 2);
      for (auto top : top_) {
        CHECK_EQ(top->tensor()->size(), expect_top_size);
      }
    } else {
      top_ = {
        create(expect_top_size, "top"),
        create(expect_top_size, "top_diff")
      };
    }
  }

  // create ops
  Op<Pool>* pool_up = create<Pool>(args_, "pool_up", "main");
  Op<PoolDown>* pool_down = create<PoolDown>(args_, "pool_down", "main");

  // forward
  B{ bottom_[0] } >> *pool_up >> B{ top_[0] };
  // backward
  B{ top_[1], top_[0], bottom_[0] } >> *pool_down >> B{ bottom_[1] };
};

}

#endif
