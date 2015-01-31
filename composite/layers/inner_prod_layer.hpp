// Copyright Lin Min 2015
#ifndef PURINE_INNER_PROD_LAYER
#define PURINE_INNER_PROD_LAYER

#include "composite/layer.hpp"
#include "operations/include/inner.hpp"
#include "operations/include/bias.hpp"
#include "operations/include/activation.hpp"

namespace purine {

class InnerProdLayer : public Layer {
 protected:
  int num_output;
 public:
  typedef vector<Blob*> B;
  typedef tuple<int> param_tuple;
  InnerProdLayer(int rank, int device, const param_tuple& args,
      const vector<Blob*>& weight = {}) : Layer(rank, device, weight) {
    std::tie(num_output) = args;
  }
  virtual ~InnerProdLayer() override {}

 protected:
  virtual void setup() override {
    CHECK(bottom_setup_);
    CHECK_EQ(bottom_.size(), 2);
    Size bottom_size = bottom_[0]->tensor()->size();
    int bottom_length = bottom_size.channels() * bottom_size.height()
        * bottom_size.width();
    Size expect_weight_size = { num_output, bottom_length, 1, 1 };
    Size expect_bias_size = { 1, num_output, 1, 1 };
    Size expect_top_size = { bottom_size.num(), num_output, 1, 1 };

    // check weight
    if (weight_.size() != 0) {
      CHECK_EQ(weight_.size(), 4);
      CHECK_EQ(weight_[0]->tensor()->size(), expect_weight_size);
      CHECK_EQ(weight_[1]->tensor()->size(), expect_bias_size);
      CHECK_EQ(weight_[2]->tensor()->size(), expect_weight_size);
      CHECK_EQ(weight_[3]->tensor()->size(), expect_bias_size);
    } else {
      weight_ = {
        create(expect_weight_size, "weight"),
        create(expect_bias_size, "bias"),
        create(expect_weight_size, "weight_diff"),
        create(expect_bias_size, "bias_diff")
      };
    }

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

    // create ops
    Op<Inner>* inner_up = create<Inner>(tuple<>(), "inner_up", "main");
    Op<InnerDown>* inner_down = create<InnerDown>(tuple<>(),
        "inner_down", "main");
    Op<InnerWeight>* inner_weight = create<InnerWeight>(tuple<>(),
        "inner_weight", "main");
    Op<Bias>* bias_up = create<Bias>(tuple<>(), "bias_up", "main");
    Op<BiasDown>* bias_down = create<BiasDown>(tuple<>(), "bias_down", "main");

    // forward
    B{ bottom_[0], weight_[0] } >> *inner_up >> B{ top_[0] };
    B{ weight_[1] } >> *bias_up >> B{ top_[0] };
    // backward
    B{ top_[1], weight_[0] } >> *inner_down >> B{ bottom_[1] };
    B{ top_[1], bottom_[0] } >> *inner_weight >> B{ weight_[2] };
    B{ top_[1] } >> *bias_down >> B{ weight_[3] };
  }
};

}

#endif
