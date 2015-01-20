// Copyright Lin Min 2015
#ifndef PURINE_COMPOSITE_GRAPH
#define PURINE_COMPOSITE_GRAPH

#include "dispatch/graph.hpp"

namespace purine {

class CompositeGraph : public Graph {
  friend CompositeGraph& operator >> (CompositeGraph& graph1,
      CompositeGraph& graph2);
  friend CompositeGraph& operator >> (const vector<Blob*>& bottom,
      CompositeGraph& graph);
  friend void operator >> (CompositeGraph& graph, const vector<Blob*>& top);
 protected:
  vector<Blob*> bottom_;
  vector<Blob*> top_;
  bool bottom_setup_ = false;
  bool top_setup_ = false;
  virtual void setup_();
 public:
  CompositeGraph(int rank = 0, int device = 0);
  virtual ~CompositeGraph() override;

  const vector<Blob*>& bottom();
  const vector<Blob*>& top();
  void set_bottom(const vector<Blob*>& bottom);
  void set_top(const vector<Blob*>& top);

  virtual vector<Blob*> bottom(int index);
  virtual vector<Blob*> top(int index);
};

}

#endif
