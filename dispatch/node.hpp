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
  virtual void compute();
  virtual void setup() override;

  inline bool is_source() const { return inputs_.size() == 0; }
  inline bool is_sink() const { return outputs_.size() == 0; }
  inline const vector<Node*>& inputs() const { return inputs_; }
  inline const vector<Node*>& outputs() const { return outputs_; }
  inline void add_input(Node* b) { inputs_.push_back(b); }
  inline void add_output(Node* b) { outputs_.push_back(b); }

  int in() const;
  int out() const;
  void inc_in();
  void inc_out();
  void clear_in();
  void clear_out();
};

}

#endif
