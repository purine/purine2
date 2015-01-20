// Copyright Lin Min 2015
#include "composite/layer.hpp"
#include "dispatch/blob.hpp"

namespace purine {

Layer::Layer(int rank, int device, const vector<Blob*>& weight)
    : CompositeGraph(rank, device) {
  weight_ = weight;
}

Layer::~Layer() {}

void Layer::setup_() {
  // cleanup
  if (top_.size() == 0 && weight_.size() == 0) {
    setup();
    return;
  }
  // if top is set from inside, clean up
  if (top_.size() != 0 && graph_name_.count(top_[0]) == 1) {
    top_.clear();
  }
  if (weight_.size() != 0 && graph_name_.count(weight_[0]) == 1) {
    weight_.clear();
  }
  this->subgraphs_.clear();
  this->graph_name_.clear();
  // setup
  setup();
}

const vector<Blob*>& Layer::weight() {
  CHECK(bottom_setup_);
  CHECK(top_setup_);
  return weight_;
}

void Layer::set_weight(const vector<Blob*>& weight) {
  weight_ = weight;
  if (bottom_setup_ && top_setup_) {
    setup_();
  }
}

vector<Blob*> Layer::weight_data() {
  return first_half<Blob*>(weight_);
}

vector<Blob*> Layer::weight_diff() {
  return second_half<Blob*>(weight_);
}

vector<Blob*> Layer::weight(int index) {
  CHECK_EQ(weight_.size() % 2, 0);
  CHECK_LT(index, weight_.size() / 2);
  int middle = weight_.size() / 2;
  return { weight_[index], weight_[middle + index] };
}

vector<Blob*> Layer::bottom_data() {
  return first_half<Blob*>(bottom_);
}

vector<Blob*> Layer::bottom_diff() {
  return second_half<Blob*>(bottom_);
}

vector<Blob*> Layer::top_data() {
  return first_half<Blob*>(top_);
}

vector<Blob*> Layer::top_diff() {
  return second_half<Blob*>(top_);
}

vector<Blob*> Layer::bottom(int index) {
  CHECK_EQ(bottom_.size() % 2, 0);
  CHECK_LT(index, bottom_.size() / 2);
  int middle = bottom_.size() / 2;
  return { bottom_[index], bottom_[middle + index] };
}

vector<Blob*> Layer::top(int index) {
  CHECK_EQ(top_.size() % 2, 0);
  CHECK_LT(index, top_.size() / 2);
  int middle = top_.size() / 2;
  return { top_[index], top_[middle + index] };
}


}
