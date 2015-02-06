// Copyright Lin Min 2015
#ifndef PURINE_LRN_LAYER
#define PURINE_LRN_LAYER

#include "composite/layer.hpp"
#include "operations/include/lrn.hpp"

namespace purine {

class LRNLayer : public Layer {
 protected:
  DTYPE alpha;
  DTYPE beta;
  int size;
 public:
  typedef tuple<DTYPE, DTYPE, int> param_tuple;
  LRNLayer(int rank, int device, const param_tuple& args)
      : Layer(rank, device) {
    std::tie(alpha, beta, size) = args;
  }
  virtual ~LRNLayer() {}

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
      top_ = {
        create("top", bottom_size),
        create("top_diff", bottom_size)
      };
    }

    // create ops
    Blob* scale = create("scale", bottom_size);
    Op<LRNScale>* lrn_scale = create<LRNScale>("lrnscale", "main",
        param_tuple(alpha, beta, size));
    Op<LRN>* lrn = create<LRN>("lrn", "main", param_tuple(alpha, beta, size));
    Op<LRNDown>* lrn_down = create<LRNDown>("lrndown", "main",
        param_tuple(alpha, beta, size));
    vector<Blob*>{ bottom_[0] } >> *lrn_scale >> vector<Blob*>{ scale };
    vector<Blob*>{ bottom_[0], scale } >> *lrn >> vector<Blob*>{ top_[0] };
    vector<Blob*>{ bottom_[0], top_[1], scale, top_[0] } >>
        *lrn_down >> vector<Blob*>{ bottom_[1] };
  }
};

}

#endif
