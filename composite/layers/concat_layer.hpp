// Copyright Lin Min 2015
#ifndef PURINE_CONCAT_LAYER
#define PURINE_CONCAT_LAYER

#include "composite/layer.hpp"
#include "composite/graph/split.hpp"
#include "composite/graph/concat.hpp"

namespace purine {

// delete set top, ConcatLayer generates top
class ConcatLayer;
const vector<Blob*>& operator >> (ConcatLayer& concat,
    const vector<Blob*>& top) = delete;

class ConcatLayer : public Layer {
 protected:
  Split::DIM dim;
 public:
  typedef tuple<Split::DIM> param_tuple;
  ConcatLayer(int rank, int device, const param_tuple& args)
      : Layer(rank, device) {
    std::tie(dim) = args;
  }
  virtual ~ConcatLayer() {}
 protected:
  virtual void setup() override {
    CHECK(bottom_setup_);

    // concat
    Concat* concat = createGraph<Concat>("concat", Concat::param_tuple(dim));
    bottom_data() >> *concat;

    // split
    top_ = {
      concat->top()[0],
      create("top_diff", concat->top()[0]->tensor()->size())
    };
    Split* split = createGraph<Split>("split", Concat::param_tuple(dim));
    top_diff() >> *split >> bottom_diff();
  }
};

}

#endif
