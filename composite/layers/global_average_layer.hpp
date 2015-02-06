// Copyright Lin Min 2015
#ifndef PURINE_GLOBAL_AVERAGE
#define PURINE_GLOBAL_AVERAGE

#include "composite/layer.hpp"
#include "operations/include/pool.hpp"

namespace purine {

class GlobalAverageLayer : public Layer {
 public:
  typedef tuple<> param_tuple;
  GlobalAverageLayer(int rank, int device, const param_tuple& args)
      : Layer(rank, device) {
  }
  virtual ~GlobalAverageLayer() override {}
 protected:
  virtual void setup() override {
    CHECK(bottom_setup_);
    CHECK_EQ(bottom_.size(), 2);
    Size bottom_size = bottom_[0]->tensor()->size();
    int kernel_h = bottom_size.height();
    int kernel_w = bottom_size.width();

    // check top
    if (top_.size() != 0) {
      CHECK_EQ(top_.size(), 2);
      for (auto top : top_) {
        CHECK_EQ(top->tensor()->size(),
            Size(bottom_size.num(), bottom_size.channels(), 1, 1));
      }
    } else {
      top_ = {
        create("top", { bottom_size.num(), bottom_size.channels(), 1, 1 }),
        create("top_diff", { bottom_size.num(), bottom_size.channels(), 1, 1 })
      };
    }

    // create ops
    Op<Pool>* pool_up = create<Pool>("pool_up", "main",
        Pool::param_tuple("average", kernel_h, kernel_w, 1, 1, 0, 0));
    Op<PoolDown>* pool_down = create<PoolDown>("pool_down", "main",
        PoolDown::param_tuple("average", kernel_h, kernel_w, 1, 1, 0, 0));

    // forward
    B{ bottom_[0] } >> *pool_up >> B{ top_[0] };
    // backward
    B{ top_[1], top_[0], bottom_[0] } >> *pool_down >> B{ bottom_[1] };
  }
};

}

#endif
