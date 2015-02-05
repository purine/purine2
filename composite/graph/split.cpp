// Copyright Lin Min 2015
#include <numeric>
#include "composite/graph/split.hpp"
#include "operations/include/dummy.hpp"

namespace purine {

void Split::setup() {
  CHECK(bottom_setup_);
  CHECK_EQ(bottom_.size(), 1);
  if (top_.size() != 0) {
    dims = vector<int>(top_.size());
    if (dim == DIM::NUM) {
      for (int i = 0; i < top_.size(); ++i) {
        dims[i] = top_[i]->tensor()->size().num();
      }
    } else {
      for (int i = 0; i < top_.size(); ++i) {
        dims[i] = top_[i]->tensor()->size().channels();
      }
    }
  } else {
    top_ = vector<Blob*>(dims.size());
    for (int i = 0; i < top_.size(); ++i) {
      top_[i] = create("top", {0, 0, 0, 0});
    }
  }
  int sum = std::accumulate(dims.begin(), dims.end(), 0);
  if (dim == DIM::NUM) {
    CHECK_EQ(sum, bottom_[0]->tensor()->size().num());
  } else {
    CHECK_EQ(sum, bottom_[0]->tensor()->size().channels());
  }

  // create sliced blobs from bottom
  Size bottom_size = bottom_[0]->tensor()->size();
  bottom_[0]->tensor()->mutable_data();
  int off = 0;
  if (dim == DIM::NUM) {
    for (int i = 0; i < dims.size(); ++i) {
      Size tmp_size = { dims[i], bottom_size.channels(), bottom_size.height(),
                        bottom_size.width() };
      Offset tmp_offset = { off, 0, 0, 0 };
      off += dims[i];
      top_[i]->tensor()->slice_from(bottom_[0]->tensor(), tmp_offset, tmp_size);
    }
  } else {
    for (int i = 0; i < dims.size(); ++i) {
      Size tmp_size = { bottom_size.num(), dims[i], bottom_size.height(),
                        bottom_size.width() };
      Offset tmp_offset = { 0, off, 0, 0 };
      off += dims[i];
      top_[i]->tensor()->slice_from(bottom_[0]->tensor(), tmp_offset, tmp_size);
    }
  }
  // create op
  Op<Dummy>* dummy = create<Dummy>("slice", "main", Dummy::param_tuple());
  bottom_ >> *dummy >> top_;
}

}
