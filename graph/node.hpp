// Copyright Lin Min 2015
#ifndef PURINE_NODE
#define PURINE_NODE

#include <atomic>
#include <vector>

#include "graph/graph.hpp"

namespace purine {

class Graph;

using std::atomic;
using std::vector;

class Node : public Graph {
 protected:
  std::atomic<int> in_;
  std::atomic<int> out_;
  vector<Node*> inputs_;
  vector<Node*> outputs_;
 public:
  explicit Node(int rank, int device);
  virtual ~Node();
  virtual void run();
  virtual void run_async();
  virtual vector<Node*> sources();
  virtual vector<Node*> sinks();
  virtual vector<Node*> nodes();
  virtual void setup();

  inline int rank() { return rank_; }
  inline int device() { return device_; }
  inline bool is_source() { return inputs_.size() == 0; }
  inline bool is_sink() { return outputs_.size() == 0; }

  int in() const;
  int out() const;
  void inc_in();
  void inc_out();
  void clear_in();
  void clear_out();
};

}

#endif
