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

vector<Blob*> Layer::weight_data() const {
  return first_half<Blob*>(weight_);
}

vector<Blob*> Layer::weight_diff() const {
  return second_half<Blob*>(weight_);
}

vector<Blob*> Layer::weight(int index) const {
  CHECK_EQ(weight_.size() % 2, 0);
  CHECK_LT(index, weight_.size() / 2);
  int middle = weight_.size() / 2;
  return { weight_[index], weight_[middle + index] };
}

}
