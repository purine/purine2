// Copyright Lin Min 2015
#ifndef PURINE_LAYER
#define PURINE_LAYER

#include "composite/connectable.hpp"

namespace purine {

class Layer : public Connectable {
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
  virtual void set_bottom(const vector<Blob*>& bottom) override;
  virtual void set_top(const vector<Blob*>& top) override;
};

}

#endif
