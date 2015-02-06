// Copyright Lin Min 2015
#ifndef PURINE_CONV_LAYER
#define PURINE_CONV_LAYER

#include "composite/layer.hpp"
#include "operations/include/conv.hpp"
#include "operations/include/bias.hpp"
#include "operations/include/activation.hpp"
#include "composite/layers/activation_layer.hpp"

namespace purine {

class ConvLayer : public Layer {
 protected:
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int kernel_h;
  int kernel_w;
  int num_output;
  string activation;
 public:
  typedef vector<Blob*> B;
  typedef tuple<int, int, int, int, int, int, int, string> param_tuple;
  ConvLayer(int rank, int device, const param_tuple& args,
      const vector<Blob*>& weight = {}) : Layer(rank, device, weight) {
    std::tie(pad_h, pad_w, stride_h, stride_w, kernel_h,
        kernel_w, num_output, activation) = args;
  }
  virtual ~ConvLayer() override {}

 protected:
  virtual void setup() override {
    CHECK(bottom_setup_);
    CHECK_EQ(bottom_.size(), 2);
    Size bottom_size = bottom_[0]->tensor()->size();
    Size expect_weight_size = {num_output, bottom_size.channels(),
                               kernel_h, kernel_w};
    Size expect_bias_size = {1, num_output, 1, 1};
    int out_h = (bottom_size.height() + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (bottom_size.width() + 2 * pad_w - kernel_w) / stride_w + 1;
    Size expect_top_size = {bottom_size.num(), num_output, out_h, out_w};

    // check weight
    if (weight_.size() != 0) { // weight is set from outside
      CHECK_EQ(weight_.size(), 4);
      CHECK_EQ(weight_[0]->tensor()->size(), expect_weight_size);
      CHECK_EQ(weight_[2]->tensor()->size(), expect_weight_size);
      CHECK_EQ(weight_[1]->tensor()->size(), expect_bias_size);
      CHECK_EQ(weight_[3]->tensor()->size(), expect_bias_size);
    } else { // generate weight
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
    Conv::param_tuple params = make_tuple(pad_h, pad_w, stride_h, stride_w);
    Op<Conv>* conv_up = create<Conv>("conv_up", "main", params);
    Op<ConvDown>* conv_down = create<ConvDown>("conv_down", "main", params);
    Op<ConvWeight>* conv_weight = create<ConvWeight>("conv_weight", "main",
        params);
    Op<Bias>* bias_up = create<Bias>("bias_up", "main", tuple<>());
    Op<BiasDown>* bias_down = create<BiasDown>("bias_down", "main", tuple<>());

    if (activation == "") {
      // forward
      B{ bottom_[0], weight_[0] } >> *conv_up >> B{ top_[0] };
      B{ weight_[1] } >> *bias_up >> B{ top_[0] };
      // backward
      B{ top_[1], weight_[0] } >> *conv_down >> B{ bottom_[1] };
      B{ top_[1], bottom_[0] } >> *conv_weight >> B{ weight_[2] };
      B{ top_[1] } >> *bias_down >> B{ weight_[3] };
    } else {
      // inplace
      Blob* tmp_data = create("...", top_[0]->shared_tensor());
      Blob* tmp_diff = create("...", top_[1]->shared_tensor());
      B{ bottom_[0], weight_[0] } >> *conv_up >> B{ tmp_data };
      B{ weight_[1] } >> *bias_up >> B{ tmp_data };
      // backward
      B{ tmp_diff, weight_[0] } >> *conv_down >> B{ bottom_[1] };
      B{ tmp_diff, bottom_[0] } >> *conv_weight >> B{ weight_[2] };
      B{ tmp_diff } >> *bias_down >> B{ weight_[3] };
      ActivationLayer* act = createGraph<ActivationLayer>("act",
          ActivationLayer::param_tuple(activation, true));
      B{ tmp_data, tmp_diff } >> *act >> top_;
    }


  }
};

}

#endif
