// Copyright Lin Min 2015
#ifndef PURINE_LAYER
#define PURINE_LAYER

#include "composite/connectable.hpp"

namespace purine {

class Layer : public Connectable {
  friend Layer& operator >> (Layer& layer1, Layer& layer2);
  friend Layer& operator >> (const vector<Blob*>& bottom, Layer& layer);
  friend const vector<Blob*>& operator >> (Layer& layer,
      const vector<Blob*>& top);
 protected:
  vector<Blob*> weight_;
  vector<Blob*> loss_;
 public:
  using Connectable::top;
  using Connectable::bottom;
  Layer(int rank = 0, int device = 0, const vector<Blob*>& weight = {});
  virtual ~Layer() override;

  const vector<Blob*>& weight() const;
  const vector<Blob*>& loss() const;

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
