// Copyright Lin Min 2015
#include <utility>
#include "composite/composite_graph.hpp"

using std::pair;

namespace purine {

/**
 * { src1, src2, ... } >> copy >> { dest1, dest2, ... }
 */
class Copy : public CompositeGraph {
  friend CompositeGraph& operator >> (Copy& copy, CompositeGraph& graph);
 protected:
  vector<int> ranks_;
  vector<int> devices_;
 public:
  typedef tuple<> param_tuple;
  Copy(const param_tuple& args, int rank, int device)
      : CompositeGraph(rank, device) {
  }
  virtual ~Copy() override {}
  void set_ranks(const vector<int>& ranks) { ranks_ = ranks; }
  void set_devices(const vector<int>& devices) { devices_ = devices; }
 protected:
  virtual void setup() override;
};

/**
 * Distribute to destination A and B
 * { src1 } >> distribute >> { destA, destB, ... }
 */
class Distribute : public CompositeGraph {
 protected:
  vector<pair<int, int> > rank_device;
 public:
  typedef tuple<vector<pair<int, int> > > param_tuple;
  Distribute(const param_tuple& args, int rank, int device)
      : CompositeGraph(rank, device) {
    std::tie(rank_device) = args;
  }
  virtual ~Distribute() {}
  virtual vector<Blob*> top(int index) override;
 protected:
  virtual void setup() override;
};

}
