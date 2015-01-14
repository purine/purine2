
#include <deque>
#include "graph/graph.hpp"

using std::deque;

namespace purine {

Graph::Graph() {
}

Graph::Graph(int rank, int device) : rank_(rank), device_(device) {
}

Graph::Graph(const vector<Graph*>& inputs, const vector<Graph*>& outputs) {
}

Graph::~Graph() {
}

vector<Node*> Graph::nodes() const {
  deque<Graph*> que;
  vector<Node*> ret;
  std::transform(subgraphs_.begin(), subgraphs_.end(), que.end(),
      [](const shared_ptr<Graph>& g)->Graph* {
        return g.get();
      });
  while (que.size() != 0) {
    Graph* g = que.front();
    que.pop_front();
    if (g->subgraphs_.size() == 0) {
      Node* n = dynamic_cast<Node*>(g);
      ret.push_back(n);
    } else {
      std::transform(g->subgraphs_.begin(), g->subgraphs_.end(), que.end(),
          [](const shared_ptr<Graph>& g)->Graph* {
            return g.get();
          });
    }
  }
  return ret;
}

vector<Node*> Graph::sources() const {
  vector<Node*>&& nodes_ = this->nodes();
  std::remove_if(nodes_.begin(), nodes_.end(),
      [](Node* n)->bool { return n->is_source() == false; });
  return nodes_;
}

vector<Node*> Graph::sinks() const {
  vector<Node*>&& nodes_ = this->nodes();
  std::remove_if(nodes_.begin(), nodes_.end(),
      [](Node* n)->bool { return n->is_sink() == false; });
  return nodes_;
}

void Graph::run() {
  vector<Node*> sources = this->sources();
  vector<Node*> sinks = this->sinks();
  for (Node* source : sources) {
    source->run();
  }
}

void Graph::run_async() {

}

void Graph::setup() {

}

void Graph::add_graph(Graph* g) {
}

Blob* Graph::create(const Size& size, const string& name, int rank,
    int device) {
  subgraphs_.push_back(shared_ptr<Graph>(new Blob(size)));
  Graph* g = subgraphs_.rbegin()->get();
  graph_name_[g] = name;
  g->parent_ = this;
  return static_cast<Blob*>(g);
}

Node::Node() : in_(0), out_(0) {
}

Node::~Node() {
}

void Node::inc_in() {
  int in = in_.fetch_add(1);
  if (in + 1 == inputs_.size()) {
    run();
    for (Node* node : inputs_) {
      node->inc_out();
    }
    clear_in();
    for (Node* node : outputs_) {
      node->inc_in();
    }
  }
}

void Node::inc_out() {
  int out = out_.fetch_add(1);
  if (out + 1 >= outputs_.size()) {
    clear_out();
  }
}

void Node::clear_in() {
  in_ = 0;
}

void Node::clear_out() {
  out_ = 0;
}

vector<Node*> Node::sources() {
  if (this->inputs_.size() == 0) {
    return { this };
  } else {
    return {};
  }
}

vector<Node*> Node::sinks() {
  if (this->outputs_.size() == 0) {
    return { this };
  } else {
    return {};
  }
}

void Node::add_input(Node* input) {
  inputs_.push_back(input);
}

void Node::add_output(Node* output) {
  outputs_.push_back(output);
}

Blob::Blob(const Size& s) {}

Blob::~Blob() {}

void Blob::run() {

  if (inputs_.size() == 0) {
    // this blob is the source of a graph.
    for (Node* out : outputs_) {
      out->inc_in();
    }
  } else {
    // this blob is triggered by its inputs.
    // pass
    // wait till this blob has the data
  }
}

Tensor* Blob::tensor() {
  return tensor_.get();
}

}
