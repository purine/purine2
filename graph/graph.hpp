#ifndef PURINE_GRAPH
#define PURINE_GRAPH

#include <algorithm>
#include <atomic>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <memory>

#include "operations/operation.hpp"
#include "operations/tensor.hpp"

namespace purine {

using std::function;
using std::map;
using std::atomic;
using std::string;
using std::vector;
using std::shared_ptr;

class Graph;
class Node;
template <typename O> class Op;
class Blob;

class Graph {
 protected:
  vector<shared_ptr<Graph> > subgraphs_;
  map<Graph*, string> graph_name_;
  Graph* parent_ = NULL;
 public:
  explicit Graph();
  explicit Graph(int rank, int device);
  explicit Graph(const vector<Graph*>& inputs,
      const vector<Graph*>& outputs = {});
  explicit Graph(Graph* input, Graph* output = NULL);
  virtual ~Graph();
  virtual void run();
  virtual void run_async();
  virtual void add_graph(Graph*);

  template <typename O>
  inline Op<O>* create(const typename O::param_tuple& param, const string& name,
      int rank, int device) {
    subgraphs_.push_back(
        shared_ptr<Graph>(new Op<O>(param, rank, device, thread)));
    Graph* g = subgraphs_.rbegin()->get();
    graph_name_[g] = name;
    g->parent_ = this;
    return static_cast<Op<O>*>(g);
  }

  template <typename O>
  inline Op<O>* create(const typename O::param_tuple& param,
      const string& name) {
    subgraphs_.push_back(
        shared_ptr<Graph>(new Op<O>(param, rank, device, thread)));
    Graph* g = subgraphs_.rbegin()->get();
    graph_name_[g] = name;
    g->parent_ = this;
    return static_cast<Op<O>*>(g);
  }

  Blob* create(const Size& size, const string& name, int rank, int device);
  virtual vector<Node*> sources() const;
  virtual vector<Node*> sinks() const;
  virtual vector<Node*> nodes() const;
  virtual void setup();
};

class Node : public Graph {
 protected:
  std::atomic<int> in_;
  std::atomic<int> out_;
  vector<Node*> inputs_;
  vector<Node*> outputs_;
 public:
  explicit Node();
  virtual ~Node();
  virtual void run() {
    LOG(FATAL) << "Not Implemented";
  }
  virtual vector<Node*> sources();
  virtual vector<Node*> sinks();
  virtual void add_input(Node* input);
  virtual void add_output(Node* output);
  inline bool is_source() { return inputs_.size() == 0; }
  inline bool is_sink() { return outputs_.size() == 0; }
  void inc_in();
  void inc_out();
  void clear_in();
  void clear_out();
};

class Blob : public Node {
 protected:
  shared_ptr<Tensor> tensor_;
 public:
  explicit Blob(const Size& s);
  virtual ~Blob();
  Tensor* tensor();
  virtual void run();
};

template <typename O>
class Op : public Node {
 protected:
  int rank_;
  int device_;
  int thread_;
  function<void()> fn_;
  typename O::param_tuple args_;
  shared_ptr<Operation> o_;
 public:
  explicit Op(const typename O::param_tuple& args,
      int rank, int device, int thread) : args_(args) {
  }
  explicit Op(const typename O::param_tuple& args,
      const initializer_list<Blob*>& inputs,
      const initializer_list<Blob*>& outputs,
      int rank, int device, int thread) : Op(args, rank, device, thread) {
    inputs_ = inputs;
    outputs_ = outputs;
  }
  virtual void run() {
    if (!fn_) {
      vector<Tensor*> input_tensors;
      vector<Tensor*> output_tensors;
      std::transform(inputs_.begin(), inputs_.end(), input_tensors.begin(),
          [] (Node* b) -> Tensor* { return static_cast<Blob*>(b)->tensor(); });
      std::transform(outputs_.begin(), outputs_.end(), output_tensors.begin(),
          [] (Node* b) -> Tensor* { return static_cast<Blob*>(b)->tensor(); });
      o_.reset(new O(input_tensors, output_tensors, args_));
    }
    // check fn_ is already set.
    // if not set fn
    // find the worker according to location and thread
    // run fn in the worker
  }
};

template <typename O>
Op<O>& operator >> (const vector<Blob*>& inputs, Op<O>& op) {
  for (Blob* input : inputs) {
    input->add_output(&op);
    op.add_input(input);
  }
  return op;
}

template <typename O>
void operator >> (Op<O>& op, const vector<Blob*>& outputs) {
  for (Blob* output : outputs) {
    output->add_input(&op);
    op.add_output(output);
  }
  return;
}

}

#endif
