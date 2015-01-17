#ifndef PURINE_GRAPH_TEMPLATE
#define PURINE_GRAPH_TEMPLATE

#include "dispatch/graph.hpp"
#include "dispatch/op.hpp"

namespace purine {

template <typename O>
Op<O>* Graph::create(const typename O::param_tuple& param, const string& name,
    int rank, int device, const string& thread) {
  subgraphs_.push_back(
      shared_ptr<Graph>(new Op<O>(param, rank, device, thread)));
  Graph* g = subgraphs_.rbegin()->get();
  graph_name_[g] = name;
  g->parent_ = this;
  return static_cast<Op<O>*>(g);
}

template <typename O>
Op<O>* Graph::create(const typename O::param_tuple& param,
    const string& name, const string& thread) {
  subgraphs_.push_back(
      shared_ptr<Graph>(new Op<O>(param, rank_, device_, thread)));
  Graph* g = subgraphs_.rbegin()->get();
  graph_name_[g] = name;
  g->parent_ = this;
  return static_cast<Op<O>*>(g);
}

}

#endif
