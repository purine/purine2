// Copyright Lin Min 2015
#include "graph/node.hpp"

namespace purine {

Node::Node(int rank, int device)
    : Graph(rank, device), in_(0), out_(0) {
}

Node::~Node() {
}

void Node::run() {
  LOG(FATAL) << "Not Implemented";
}

void Node::run_async() {
  LOG(FATAL) << "Not Implemented";
}

vector<Node*> Node::sources() {
  if (this->inputs_.size() == 0 && rank_ == current_rank()) {
    return { this };
  } else {
    return {};
  }
}

vector<Node*> Node::sinks() {
  if (this->outputs_.size() == 0 && rank_ == current_rank()) {
    return { this };
  } else {
    return {};
  }
}

vector<Node*> Node::nodes() {
  if (rank_ == current_rank()) {
    return { this };
  } else {
    return {};
  }
}

int Node::in() const {
  return in_;
}

int Node::out() const {
  return out_;
}

void Node::setup() {
}

// ???
void Node::inc_in() {
  int in = in_.fetch_add(1);
  if (in + 1 == inputs_.size()) {
    run();
    for (Node* node : inputs_) {
      node->inc_out();
    }
    clear_in();
    // for (Node* node : outputs_) {
    //   node->inc_in();
    // }
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

}
