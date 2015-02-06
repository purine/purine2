// Copyright Lin Min 2015
#ifndef PURINE_INCEPTION_LAYER
#define PURINE_INCEPTION_LAYER

#include "composite/layer.hpp"
#include "composite/layers/pool_layer.hpp"
#include "composite/layers/conv_layer.hpp"
#include "composite/layers/concat_layer.hpp"

namespace purine {

// Inception generates top
class InceptionLayer;
const vector<Blob*>& operator >> (InceptionLayer& inception,
    const vector<Blob*>& top) = delete;

class InceptionLayer : public Layer {
 protected:
  int one;
  int three;
  int five;
  int three_reduce;
  int five_reduce;
  int pool_proj;
 public:
  typedef vector<Blob*> B;
  typedef tuple<int, int, int, int, int, int> param_tuple;
  InceptionLayer(int rank, int device, const param_tuple& args)
      : Layer(rank, device) {
    std::tie(one, three, five, three_reduce, five_reduce, pool_proj) = args;
  }
  virtual ~InceptionLayer() override {}

 protected:
  virtual void setup() override {
    CHECK(bottom_setup_);
    CHECK_EQ(bottom_.size(), 2);
    Size bottom_size = bottom_[0]->tensor()->size();

    ConvLayer* one_ = createGraph<ConvLayer>("one",
        ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, one, ""));
    ConvLayer* three_reduce_ = createGraph<ConvLayer>("three_reduce",
        ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, three_reduce, "relu"));
    ConvLayer* five_reduce_ = createGraph<ConvLayer>("five_reduce",
        ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, five_reduce, "relu"));
    ConvLayer* three_ = createGraph<ConvLayer>("three",
        ConvLayer::param_tuple(1, 1, 1, 1, 3, 3, three, ""));
    ConvLayer* five_ = createGraph<ConvLayer>("five",
        ConvLayer::param_tuple(2, 2, 1, 1, 5, 5, five, ""));
    PoolLayer* max_pool_ = createGraph<PoolLayer>("max_pool",
        PoolLayer::param_tuple("max", 3, 3, 1, 1, 1, 1));
    ConvLayer* pool_proj_ = createGraph<ConvLayer>("pool_proj",
        ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, pool_proj, ""));
    ConcatLayer* concat = createGraph<ConcatLayer>("concat",
        ConcatLayer::param_tuple(Split::CHANNELS));
    ActivationLayer* act = createGraph<ActivationLayer>("act",
        ActivationLayer::param_tuple("relu", true));

    bottom_ >> *one_;
    bottom_ >> *three_reduce_ >> *three_;
    bottom_ >> *five_reduce_ >> *five_;
    bottom_ >> *max_pool_ >> *pool_proj_;

    vector<Blob*>{ one_->top()[0], three_->top()[0], five_->top()[0],
          pool_proj_->top()[0], one_->top()[1], three_->top()[1],
          five_->top()[1], pool_proj_->top()[1] } >> *concat >> *act;
    top_ = act->top();

    vector<Layer*> layers = { one_, three_reduce_, five_reduce_,
                              three_, five_, pool_proj_ };
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
