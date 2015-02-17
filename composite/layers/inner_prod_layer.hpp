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
  string activation;
 public:
  typedef vector<Blob*> B;
  typedef tuple<int, string> param_tuple;
  InnerProdLayer(int rank, int device, const param_tuple& args,
      const vector<Blob*>& weight = {}) : Layer(rank, device, weight) {
    std::tie(num_output, activation) = args;
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
        create("weight", expect_weight_size),
        create("bias", expect_bias_size),
        create("weight_diff", expect_weight_size),
        create("bias_diff", expect_bias_size)
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
        create("top", expect_top_size),
        create("top_diff", expect_top_size)
      };
    }

    // create ops
    Op<Inner>* inner_up = create<Inner>("inner_up", "main", tuple<>());
    Op<InnerDown>* inner_down = create<InnerDown>("inner_down", "main",
        tuple<>());
    Op<InnerWeight>* inner_weight = create<InnerWeight>("inner_weight", "main",
        tuple<>());
    Op<Bias>* bias_up = create<Bias>("bias_up", "main", tuple<>());
    Op<BiasDown>* bias_down = create<BiasDown>("bias_down", "main", tuple<>());

    if (activation == "") {
      // forward
      B{ bottom_[0], weight_[0] } >> *inner_up >> B{ top_[0] };
      B{ weight_[1] } >> *bias_up >> B{ top_[0] };
      // backward
      B{ top_[1], weight_[0] } >> *inner_down >> B{ bottom_[1] };
      B{ top_[1], bottom_[0] } >> *inner_weight >> B{ weight_[2] };
      B{ top_[1] } >> *bias_down >> B{ weight_[3] };
    } else {
      // inplace
      Blob* tmp_data = create("before_act", top_[0]->shared_tensor());
      Blob* tmp_diff = create("before_act_diff", top_[1]->shared_tensor());
      B{ bottom_[0], weight_[0] } >> *inner_up >> B{ tmp_data };
      B{ weight_[1] } >> *bias_up >> B{ tmp_data };
      // backward
      B{ tmp_diff, weight_[0] } >> *inner_down >> B{ bottom_[1] };
      B{ tmp_diff, bottom_[0] } >> *inner_weight >> B{ weight_[2] };
      B{ tmp_diff } >> *bias_down >> B{ weight_[3] };
      ActivationLayer* act = createGraph<ActivationLayer>("act",
          ActivationLayer::param_tuple(activation, true));
      B{ tmp_data, tmp_diff } >> *act >> top_;
    }
  }
};

}

#endif
