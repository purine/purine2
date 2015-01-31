// Copyright Lin Min 2015
#include <utility>
#include "dispatch/graph_template.hpp"
#include "composite/connectable.hpp"

using std::pair;

namespace purine {

/**
 * { src } >> copy >> { dest }
 * copy's rank_ and device_ denotes the output location
 */
class Copy : public Connectable {
  friend Connectable& operator >> (Copy& copy, Connectable& graph);
  friend Copy& operator >> (Connectable& graph, Copy& copy);
  friend Copy& operator >> (Copy& copy1, Copy& copy2);
  friend Copy& operator >> (const vector<Blob*>& inputs, Copy& copy);
  friend const vector<Blob*>& operator >> (Copy& copy,
      const vector<Blob*>& outputs);
 public:
  typedef tuple<> param_tuple;
  Copy(int rank = 0, int device = 0)
      : Connectable(rank, device) {
  }
  virtual ~Copy() override {}
 protected:
  virtual void setup() override;
};

/**
 * Distribute to destination A and B
 * { src1 } >> distribute >> { destA, destB, ... }
 */
class Distribute : public Connectable {
 protected:
  vector<pair<int, int> > rank_device;
 public:
  typedef tuple<vector<pair<int, int> > > param_tuple;
  Distribute(int rank, int device, const param_tuple& args)
      : Connectable(rank, device) {
    std::tie(rank_device) = args;
  }
  virtual ~Distribute() override {}
 protected:
  virtual void setup() override;
};

/**
 * { blob1, blob2, ... } >> agg >> { dest }
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
  typedef tuple<Type> param_tuple;
  Aggregate(int rank, int device, Aggregate::Type type)
      : Connectable(rank, device) {
    agg_type_ = type;
  }
  virtual ~Aggregate() override {}
};

}
