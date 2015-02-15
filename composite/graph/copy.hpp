// Copyright Lin Min 2015

#ifndef PURINE_COPY
#define PURINE_COPY

#include <utility>
#include "dispatch/graph_template.hpp"
#include "composite/connectable.hpp"

using std::pair;

namespace purine {

class Copy;
Copy& operator >> (Copy& copy1, Copy& copy2) = delete;

/**
 * { src } >> copy >> { dest }
 * copy's rank_ and device_ denotes the output location
 * rank and device are not required
 */
class Copy : public Connectable {
  // friend Connectable& operator >> (Copy& copy, Connectable& graph);
  // friend Copy& operator >> (Connectable& graph, Copy& copy);
  // friend Copy& operator >> (const vector<Blob*>& inputs, Copy& copy);
  // friend const vector<Blob*>& operator >> (Copy& copy,
  //     const vector<Blob*>& outputs);
 public:
  typedef tuple<int, int> param_tuple;
  Copy(const param_tuple& args) {
    std::tie(rank_, device_) = args;
  }
  virtual ~Copy() override {}
 protected:
  virtual void setup() override;
};

/**
 * Distribute to destination A and B
 * { src1 } >> distribute >> { destA, destB, ... }
 * rank and device are not required
 */
class Distribute : public Connectable {
 protected:
  vector<pair<int, int> > rank_device;
 public:
  typedef tuple<vector<pair<int, int> > > param_tuple;
  Distribute(const param_tuple& args) {
    std::tie(rank_device) = args;
  }
  virtual ~Distribute() override {}
 protected:
  virtual void setup() override;
};

/**
 * { blob1, blob2, ... } >> agg >> { dest }
 * rank and device are not required.
 */
class Aggregate : public Connectable {
 public:
  enum Type {
    SUM,
    AVERAGE
  };
 protected:
  Type agg_type_;
  virtual void setup() override;
 public:
  typedef tuple<Type, int, int> param_tuple;
  Aggregate(const param_tuple& args) {
    std::tie(agg_type_, rank_, device_) = args;
  }
  virtual ~Aggregate() override {}
};

}

#endif
