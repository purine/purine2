// Copyright Lin Min 2015
#ifndef PURINE_NIN_LAYER
#define PURINE_NIN_LAYER

#include "composite/layer.hpp"
#include "composite/layers/conv_layer.hpp"

namespace purine {

class NINLayer : public Layer {
 protected:
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int kernel_h;
  int kernel_w;
  string activation;
  vector<int> channels;
 public:
  typedef vector<Blob*> B;
  typedef tuple<int, int, int, int, int, int, string, vector<int> > param_tuple;
  NINLayer(int rank, int device, const param_tuple& args)
      : Layer(rank, device) {
    std::tie(pad_h, pad_w, stride_h, stride_w, kernel_h, kernel_w, activation,
        channels) = args;
  }
  virtual ~NINLayer() override {}

 protected:
  virtual void setup() override {
    CHECK(bottom_setup_);
    CHECK_EQ(bottom_.size(), 2);
    Size bottom_size = bottom_[0]->tensor()->size();

    ConvLayer* conv = createGraph<ConvLayer>("one",
        ConvLayer::param_tuple(pad_h, pad_w, stride_h, stride_w, kernel_h,
            kernel_w, channels[0], activation));
    bottom_ >> *conv;
    vector<Layer*> layers = { conv };

    ConvLayer* prev = conv;
    for (int i = 1; i < channels.size(); ++i) {
      ConvLayer* cccp = createGraph<ConvLayer>("cccp",
          ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, channels[i], activation));
      *prev >> *cccp;
      prev = cccp;
      layers.push_back(cccp);
    }
    top_ = prev->top();

    for (auto layer : layers) {
      const vector<Blob*>& w = layer->weight_data();
      weight_.insert(weight_.end(), w.begin(), w.end());
    }
    for (auto layer : layers) {
      const vector<Blob*>& w = layer->weight_diff();
      weight_.insert(weight_.end(), w.begin(), w.end());
    }
  }
};

}

#endif
