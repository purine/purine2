// Copyright Lin Min 2015
#include "dispatch/blob.hpp"
#include "composite/connectable.hpp"
#include "composite/graph/copy.hpp"
#include "composite/vectorize.hpp"

namespace purine {

typedef vector<Blob*> B;

Connectable::Connectable(int rank, int device)
    : Graph(rank, device) {
}

Connectable::~Connectable() {}

const vector<Blob*>& Connectable::bottom() {
  CHECK(bottom_setup_);
  return bottom_;
}

const vector<Blob*>& Connectable::top() {
  CHECK(bottom_setup_);
  if (top_setup_) {
    return top_;
  } else {
    // top is not set yet, will be set in setup().
    setup(); // setup the layer.
    top_setup_ = true;
    return top_;
  }
}

void Connectable::set_top(const vector<Blob*>& top) {
  CHECK(!top_setup_);
  top_ = top;
  top_setup_ = true;
  if (bottom_setup_) { // if bottom and weight are already setup
    setup(); // setup the layer
  }
}

void Connectable::set_bottom(const vector<Blob*>& bottom) {
  CHECK(!bottom_setup_);
  bottom_ = bottom;
  bottom_setup_ = true;
  if (top_setup_) { // if top is already setup
    setup(); // setup the layer
  }
}

Connectable& operator >> (Connectable& graph1, Connectable& graph2) {
  graph2.set_bottom(graph1.top());
  return graph2;
}

Connectable& operator >> (const vector<Blob*>& bottom,
    Connectable& graph) {
  graph.set_bottom(bottom);
  return graph;
}

const vector<Blob*>& operator >> (Connectable& graph,
    const vector<Blob*>& top) {
  graph.set_top(top);
  return top;
}

}
