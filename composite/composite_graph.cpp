// Copyright Lin Min 2015
#include "dispatch/blob.hpp"
#include "composite/composite_graph.hpp"

namespace purine {

CompositeGraph::CompositeGraph(int rank, int device)
    : Graph(rank, device) {
}

CompositeGraph::~CompositeGraph() {}

void CompositeGraph::setup_() {
  // cleanup
  if (top_.size() == 0) {
    setup();
    return;
  }
  // if top is set from inside, clean up
  if (graph_name_.count(top_[0]) == 1) {
    top_.clear();
  }
  this->subgraphs_.clear();
  this->graph_name_.clear();
  // setup
  setup();
}

const vector<Blob*>& CompositeGraph::bottom() {
  CHECK(bottom_setup_);
  return bottom_;
}

const vector<Blob*>& CompositeGraph::top() {
  CHECK(bottom_setup_);
  if (top_setup_) {
    return top_;
  } else {
    // top is not set yet, will be set in setup_().
    setup_(); // setup the layer.
    top_setup_ = true;
    return top_;
  }
}

void CompositeGraph::set_top(const vector<Blob*>& top) {
  top_ = top;
  top_setup_ = true;
  if (bottom_setup_) { // if bottom and weight are already setup
    setup_(); // setup the layer
  }
}

void CompositeGraph::set_bottom(const vector<Blob*>& bottom) {
  bottom_ = bottom;
  bottom_setup_ = true;
  if (top_setup_) { // if top is already setup
    setup_(); // setup the layer
  }
}

vector<Blob*> CompositeGraph::bottom(int index) {
  CHECK_LT(index, bottom_.size());
  return { bottom_[index] };
}

vector<Blob*> CompositeGraph::top(int index) {
  CHECK_LT(index, top_.size());
  return { top_[index] };
}

CompositeGraph& operator >> (const vector<Blob*>& bottom,
    CompositeGraph& g) {
  g.set_bottom(bottom);
  return g;
}

void operator >> (CompositeGraph& g, const vector<Blob*>& top) {
  g.set_top(top);
}

CompositeGraph& operator >> (CompositeGraph& g1, CompositeGraph& g2) {
  return g1.top() >> g2;
}

}
