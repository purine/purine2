// Copyright Lin Min 2015
#ifndef PURINE_NODE
#define PURINE_NODE

#include <atomic>
#include <vector>

#include "dispatch/graph.hpp"

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
  explicit Node(int rank = 0, int device = 0);
  virtual ~Node() override;
  virtual void run() override;
  virtual void run_async() override;
  virtual vector<Node*> sources() override;
  virtual vector<Node*> sinks() override;
  virtual vector<Node*> nodes() override;
  virtual void setup() override;

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
