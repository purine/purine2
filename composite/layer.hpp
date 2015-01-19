// Copyright Lin Min 2015
#ifndef PURINE_LAYER
#define PURINE_LAYER

#include "composite/composite_graph.hpp"

namespace purine {

class Layer : public CompositeGraph {
 protected:
  vector<Blob*> weight_;
  virtual void setup_();
 public:
  Layer(int rank = 0, int device = 0, const vector<Blob*>& weight = {});
  virtual ~Layer();

  const vector<Blob*>& weight();
  void set_weight(const vector<Blob*>& weight);

  vector<Blob*> weight_data() const;
  vector<Blob*> weight_diff() const;
  vector<Blob*> weight(int index) const;
};

}

#endif
