// Copyright Lin Min 2015
#ifndef PURINE_SPLIT_LAYER
#define PURINE_SPLIT_LAYER

#include "composite/layer.hpp"
#include "composite/graph/split.hpp"
#include "composite/graph/concat.hpp"

namespace purine {

// delete set top, SplitLayer generates top
class SplitLayer;
const vector<Blob*>& operator >> (SplitLayer& split,
    const vector<Blob*>& top) = delete;

class SplitLayer : public Layer {
 protected:
  Split::DIM dim;
  vector<int> dims;
 public:
  typedef tuple<Split::DIM, vector<int> > param_tuple;
  SplitLayer(int rank, int device, const param_tuple& args)
      : Layer(rank, device) {
    std::tie(dim, dims) = args;
  }
  virtual ~SplitLayer() {}
 protected:
  virtual void setup() override {
    CHECK(bottom_setup_);
    CHECK_EQ(bottom_.size(), 2);
    Size bottom_size = bottom_[0]->tensor()->size();
    Split* split = createGraph<Split>("split", dim, dims);
    bottom_data() >> *split;
    top_.insert(top_.end(), split->top().begin(), split->top().end());
    // concat
    // create top_diff
    for (Blob* top : split->top()) {
      top_.push_back(create(top->tensor()->size(), "top_diff"));
    }
    Concat* concat = createGraph<Concat>("concat", dim);
    top_diff() >> *concat >> bottom_diff();
  }
};

}
