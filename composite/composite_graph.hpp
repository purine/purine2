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
  virtual void setup() = 0;
 public:
  CompositeGraph(int rank = 0, int device = 0);
  virtual ~CompositeGraph();

  const vector<Blob*>& bottom();
  const vector<Blob*>& top();
  void set_bottom(const vector<Blob*>& bottom);
  void set_top(const vector<Blob*>& top);

  vector<Blob*> bottom_data() const;
  vector<Blob*> bottom_diff() const;
  vector<Blob*> bottom(int index) const;

  vector<Blob*> top_data() const;
  vector<Blob*> top_diff() const;
  vector<Blob*> top(int index) const;

};

}

#endif
