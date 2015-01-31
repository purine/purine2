// Copyright Lin Min 2015
#ifndef PURINE_SPLIT
#define PURINE_SPLIT

#include <utility>
#include "dispatch/graph_template.hpp"
#include "composite/connectable.hpp"

using std::pair;

namespace purine {

/**
 * { src } >> slice >> { dest1, dest2, dest3, ... }
 */
class Split : public Connectable {
 public:
  enum DIM {
    NUM = 0,
    CHANNELS = 1
  };
 protected:
  DIM dim;
  vector<int> dims;
 public:
  typedef tuple<DIM> param_tuple;
  Split(int rank, int device, DIM d, const vector<int>& ds = {})
      : Connectable(rank, device), dim(d), dims(ds) {
  }
  inline void set_dims(const vector<int> dims) { this->dims = dims; }
  virtual ~Split() override {}
 protected:
  virtual void setup() override;
};

}

#endif
