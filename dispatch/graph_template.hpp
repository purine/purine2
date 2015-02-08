#ifndef PURINE_GRAPH_TEMPLATE
#define PURINE_GRAPH_TEMPLATE

#include "dispatch/graph.hpp"
#include "dispatch/op.hpp"
#include "dispatch/op_template.hpp"
// #include "composite/connectable.hpp"
// #include "composite/layer.hpp"

namespace purine {

class Connectable;
class Layer;

template <typename O>
Op<O>* Graph::create(const string& name, int rank, int device,
    const string& thread, const typename O::param_tuple& param) {
  subgraphs_.push_back(
      shared_ptr<Graph>(new Op<O>(rank, device, thread, param)));
  Graph* g = subgraphs_.rbegin()->get();
  graph_name_[g] = name;
  g->parent_ = this;
  return static_cast<Op<O>*>(g);
}

template <typename O>
Op<O>* Graph::create(const string& name, const string& thread,
    const typename O::param_tuple& param) {
  return create<O>(name, rank_, device_, thread, param);
}

template <typename G, typename... Args>
G* Graph::createGraph(const string& name, const Args&... args) {
  subgraphs_.push_back(shared_ptr<Graph>(new G(rank_, device_, args...)));
  Graph* g = subgraphs_.rbegin()->get();
  graph_name_[g] = name;
  g->parent_ = this;
  return static_cast<G*>(g);
}

template <typename G, typename... Args>
G* Graph::createGraph(const string& name, int rank, int device,
    const Args&... args) {
  subgraphs_.push_back(shared_ptr<Graph>(new G(rank, device, args...)));
  Graph* g = subgraphs_.rbegin()->get();
  graph_name_[g] = name;
  g->parent_ = this;
  return static_cast<G*>(g);
}

template <typename G, typename... Args>
G* Graph::createAny(const string& name, const Args&... args) {
  subgraphs_.push_back(shared_ptr<Graph>(new G(args...)));
  Graph* g = subgraphs_.rbegin()->get();
  graph_name_[g] = name;
  g->parent_ = this;
  return static_cast<G*>(g);
}

}

#endif
