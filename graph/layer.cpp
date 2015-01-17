// Copyright Lin Min 2015
#include "graph/layer.hpp"

namespace purine {

Layer::Layer(const vector<Blob*>& bottom, const vector<Blob*>& weight,
    const vector<Blob*>& top, int rank, int device) : Graph(rank, device) {
  set_bottom_weight(bottom, weight);
  set_top(top);
  setup_();
}

Layer::Layer(const vector<Blob*>& bottom, const vector<Blob*>& weight,
    int rank, int device) : Graph(rank, device) {
  set_bottom_weight(bottom, weight);
  setup_();
}

Layer::Layer(int rank, int device) : Graph(rank, device) {
}

Layer::~Layer() {}

void Layer::setup_() {
  // cleanup
  this->subgraphs_.clear();
  // setup
  setup();
}

const vector<Blob*>& Layer::bottom() {
  CHECK(bottom_setup_);
  return bottom_;
}

const vector<Blob*>& Layer::top() {
  if (top_setup_) {
    return top_;
  } else {
    // top is not set yet, will be set in setup_().
    setup_(); // setup the layer.
    top_setup_ = true;
    return top_;
  }
}

const vector<Blob*>& Layer::weight() {
  CHECK(bottom_setup_);
  return weight_;
}

void Layer::set_bottom_weight(const vector<Blob*>& bottom,
    const vector<Blob*>& weight) {
  bottom_ = bottom;
  weight_ = weight;
  bottom_setup_ = true;
  if (top_setup_) { // if top is already setup
    setup_(); // setup the layer
  }
}

void Layer::set_top(const vector<Blob*>& top) {
  top_ = top;
  top_setup_ = true;
  if (bottom_setup_) { // if bottom and weight are already setup
    setup_(); // setup the layer
  }
}

Layer& operator >> (const vector<Blob*>& bottom, Layer& layer) {
  layer.set_bottom_weight(bottom);
  return layer;
}

Layer& operator >> (const vector<vector<Blob*> >& bottom_weight, Layer& layer) {
  CHECK_EQ(bottom_weight.size(), 2); // one for bottom, one for weight
  layer.set_bottom_weight(bottom_weight[0], bottom_weight[1]);
  return layer;
}

void operator >> (Layer& layer, const vector<Blob*>& top) {
  layer.set_top(top);
}

Layer& operator >> (Layer& layer1, Layer& layer2) {
  return layer1.top() >> layer2;
}

}
