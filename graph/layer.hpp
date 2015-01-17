// Copyright Lin Min 2015
#ifndef PURINE_LAYER
#define PURINE_LAYER

#include "dispatch/graph.hpp"

namespace purine {

class Layer : public Graph {
  friend Layer& operator >> (Layer& layer1, Layer& layer2);
  friend Layer& operator >> (const vector<Blob*>& bottom, Layer& layer);
  friend Layer& operator >> (const vector<vector<Blob*> >& bottom_weight,
      Layer& layer);
  friend void operator >> (Layer& layer, const vector<Blob*>& top);
 protected:
  vector<Blob*> bottom_;
  vector<Blob*> top_;
  vector<Blob*> weight_;
  bool bottom_setup_ = false;
  bool top_setup_ = false;
  void setup_();
  virtual void setup() = 0;
 public:
  Layer(const vector<Blob*>& bottom, const vector<Blob*>& weight,
      const vector<Blob*>& top, int rank = 0, int device = 0);
  Layer(const vector<Blob*>& bottom, const vector<Blob*>& weight = {},
      int rank = 0, int device = 0);
  Layer(int rank = 0, int device = 0);
  virtual ~Layer();

  const vector<Blob*>& bottom();
  const vector<Blob*>& top();
  const vector<Blob*>& weight();
  void set_bottom_weight(const vector<Blob*>& bottom,
      const vector<Blob*>& weight = {});
  void set_top(const vector<Blob*>& top);
};

}

#endif
