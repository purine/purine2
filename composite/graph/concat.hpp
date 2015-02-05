// Copyright Lin Min 2015
#ifndef PURINE_CONCAT
#define PURINE_CONCAT

#include <utility>
#include "dispatch/graph_template.hpp"
#include "composite/connectable.hpp"
#include "composite/graph/split.hpp"

using std::pair;

namespace purine {

/**
 * { src1, src2, src3, ... } >> concat >> { dest }
 * rank and device are not required
 */
class Concat : public Connectable {
  friend const Concat& operator >> (const vector<Blob*>& inputs,
      Concat& concat);
 protected:
  Split::DIM dim;
 public:
  typedef tuple<Split::DIM> param_tuple;
  Concat(int rank, int device, const param_tuple& args)
      : Connectable(rank, device) {
    std::tie(dim) = args;
  }
  virtual ~Concat() override {}
 protected:
  virtual void setup() override;
};

}

#endif
