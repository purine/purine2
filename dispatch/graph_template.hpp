#ifndef PURINE_GRAPH_TEMPLATE
#define PURINE_GRAPH_TEMPLATE

#include "dispatch/graph.hpp"
#include "dispatch/op.hpp"
#include "dispatch/op_template.hpp"

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
  return create<O>(param, name, rank_, device_, thread);
}

template <typename G>
G* Graph::create(const typename G::param_tuple& param, const string& name,
    int rank, int device) {
  subgraphs_.push_back(shared_ptr<Graph>(new G(param, rank, device)));
  Graph* g = subgraphs_.rbegin()->get();
  graph_name_[g] = name;
  g->parent_ = this;
  return static_cast<G*>(g);
}

template <typename G>
G* Graph::create(const typename G::param_tuple& param,
    const string& name) {
  return create<G>(param, name, rank_, device_);
}

}

#endif
