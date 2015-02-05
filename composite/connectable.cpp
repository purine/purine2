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

vector<Blob*> Connectable::bottom(int index) {
  CHECK_LT(index, bottom_.size());
  return { bottom_[index] };
}

vector<Blob*> Connectable::top(int index) {
  CHECK_LT(index, top_.size());
  return { top_[index] };
}

// Connectables, when connect to other connectables,
// will place a Copy in the middle. So that blobs on different devices or ranks
// will be moved to the correct place.
// Except Copy which is also a Connectable. But the operators are overrided.

Connectable& operator >> (const vector<Blob*>& bottom, Connectable& g) {
  (vector<B>{ bottom } >> *g.createFlexible<Vectorize<Copy> >("...",
      vector<Copy::param_tuple>(bottom.size(),
          Copy::param_tuple(g.rank(), g.device())))).top()[0] >> g;
  // CHECK that inputs are on the same location as g
  if (!g.FLEXIBLE()) {
    for (Blob* b : bottom) {
      CHECK_EQ(b->rank(), g.rank());
      CHECK_EQ(b->device(), g.device());
    }
  }
  return g;
}

const vector<Blob*>& operator >> (Connectable& g, const vector<Blob*>& top) {
  vector<Blob*> top_tmp(top.size());
  transform(top.begin(), top.end(), top_tmp.begin(),
      [&](Blob* b)->Blob* {
        if (b->rank() == g.rank() && b->device() == g.device()) {
          return b;
        } else {
          Blob* tmp = g.create("...", g.rank(), g.device(),
              b->tensor()->size());
          B{ tmp } >> *g.createFlexible<Copy>("...", Copy::param_tuple())
                          >> B{ b };
          return tmp;
        }
      });
  g.set_top(top_tmp);
  // CHECK top are on the same location as g
  if (!g.FLEXIBLE()) {
    for (Blob* t : top) {
      CHECK_EQ(t->rank(), g.rank());
      CHECK_EQ(t->device(), g.device());
    }
  }
  return top;
}

Connectable& operator >> (Connectable& g1, Connectable& g2) {
  return g1.top() >> g2;
}

}
