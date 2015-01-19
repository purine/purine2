// Copyright Lin Min 2015
#ifndef PURINE_CONV_LAYER
#define PURINE_CONV_LAYER

#include "composite/layer.hpp"
#include "operations/include/conv.hpp"
#include "operations/include/bias.hpp"
#include "operations/include/activation.hpp"

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
 public:
  typedef vector<Blob*> B;
  typedef tuple<int, int, int, int, int, int, int> param_tuple;
  ConvLayer(const param_tuple& args, int rank, int device,
      const vector<Blob*>& weight = {}) : Layer(rank, device, weight) {
    std::tie(pad_h, pad_w, stride_h, stride_w, kernel_h,
        kernel_w, num_output) = args;
  }
  virtual ~ConvLayer() {}

  virtual void setup() {
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
    Conv::param_tuple params = make_tuple(pad_h, pad_w, stride_h, stride_w);
    Op<Conv>* conv_up = create<Conv>(params, "conv_up", "main");
    Op<ConvDown>* conv_down = create<ConvDown>(params, "conv_down", "main");
    Op<ConvWeight>* conv_weight = create<ConvWeight>(params,
        "conv_weight", "main");
    Op<Bias>* bias_up = create<Bias>(tuple<>(), "bias_up", "main");
    Op<BiasDown>* bias_down = create<BiasDown>(tuple<>(), "bias_down", "main");

    // forward
    B{ bottom_[0], weight_[0] } >> *conv_up >> B{ top_[0] };
    B{ weight_[1] } >> *bias_up >> B{ top_[0] };
    // backward
    B{ top_[1], weight_[0] } >> *conv_down >> B{ bottom_[1] };
    B{ top_[1], bottom_[0] } >> *conv_weight >> B{ weight_[2] };
    B{ top_[1] } >> *bias_down >> B{ weight_[3] };
  }
};

}

#endif
