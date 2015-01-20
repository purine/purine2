// Copyright Lin Min 2015
#ifndef PURINE_AGGREGATE
#define PURINE_AGGREGATE

#include "dispatch/graph.hpp"
#include "dispatch/graph_template.hpp"
#include "dispatch/op.hpp"
#include "dispatch/op_template.hpp"
#include "dispatch/blob.hpp"
#include "composite/composite_graph.hpp"

class Aggregate : public CompositeGraph {
 protected:
  Type agg_type_;
  virtual void setup() override {
  }
 public:
  enum Type {
    SUM,
    AVERAGE,
  };
  Aggregate() : agg_type_(Type::SUM) {
  }
  virtual ~Aggregate() override;
};

#endif
