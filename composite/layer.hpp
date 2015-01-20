// Copyright Lin Min 2015
#ifndef PURINE_LAYER
#define PURINE_LAYER

#include "composite/composite_graph.hpp"

namespace purine {

class Layer : public CompositeGraph {
 protected:
  vector<Blob*> weight_;
  virtual void setup_() override;
 public:
  using CompositeGraph::top;
  using CompositeGraph::bottom;
  Layer(int rank = 0, int device = 0, const vector<Blob*>& weight = {});
  virtual ~Layer() override;

  const vector<Blob*>& weight();
  void set_weight(const vector<Blob*>& weight);

  vector<Blob*> weight_data();
  vector<Blob*> weight_diff();
  vector<Blob*> bottom_data();
  vector<Blob*> bottom_diff();
  vector<Blob*> top_data();
  vector<Blob*> top_diff();

  virtual vector<Blob*> weight(int index);
  virtual vector<Blob*> bottom(int index) override;
  virtual vector<Blob*> top(int index) override;
};

}

#endif
