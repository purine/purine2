// Copyright Lin Min 2015
#include "composite/graph/concat.hpp"
#include "operations/include/dummy.hpp"

namespace purine {

void Concat::setup() {
  CHECK(bottom_setup_);
  CHECK_GT(bottom_.size(), 1);
  // check bottom size
  Size s = bottom_[0]->tensor()->size();
  Size expected_top_size;
  int sum = 0;
  if (dim == Split::DIM::NUM) {
    for (Blob* b : bottom_) {
      CHECK_EQ(b->tensor()->size().channels(), s.channels());
      CHECK_EQ(b->tensor()->size().height(), s.height());
      CHECK_EQ(b->tensor()->size().width(), s.width());
      sum += b->tensor()->size().num();
    }
    expected_top_size = { sum, s.channels(), s.height(), s.width() };
  } else {
    for (Blob* b : bottom_) {
      CHECK_EQ(b->tensor()->size().num(), s.num());
      CHECK_EQ(b->tensor()->size().height(), s.height());
      CHECK_EQ(b->tensor()->size().width(), s.width());
      sum += b->tensor()->size().channels();
    }
    expected_top_size = { s.num(), sum, s.height(), s.width() };
  }

  // check top
  if (top_.size() != 0) {
    CHECK_EQ(expected_top_size, top_[0]->tensor()->size());
  } else {
    top_ = {
      create(expected_top_size, "top")
    };
  }
  top_[0]->tensor()->mutable_data();

  // create sliced blobs from bottom
  int off = 0;
  if (dim == Split::DIM::NUM) {
    for (int i = 0; i < bottom_.size(); ++i) {
      top_[i]->tensor()->slice_from(top_[0]->tensor(), { off, 0, 0, 0},
          bottom_[i]->tensor()->size());
      off += bottom_[i]->tensor()->size().num();
    }
  } else {
    for (int i = 0; i < bottom_.size(); ++i) {
      top_[i]->tensor()->slice_from(top_[0]->tensor(), { 0, off, 0, 0},
          bottom_[i]->tensor()->size());
      off += bottom_[i]->tensor()->size().channels();
    }
  }
  // create op
  Op<Dummy>* dummy = create<Dummy>(Dummy::param_tuple(), "concat", "main");
  bottom_ >> *dummy >> top_;
}

}
