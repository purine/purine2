// Copyright Lin Min 2015
#include <deque>
#include <iterator>
#include <string>
#include "dispatch/graph.hpp"
#include "dispatch/node.hpp"
#include "dispatch/blob.hpp"

using std::deque;

namespace purine {

Graph::Graph(int rank, int device) : rank_(rank), device_(device) {
}

Graph::~Graph() {
}

/**
 * @brief return all nodes in this graph, including nodes in the subgraph.
 */
vector<Node*> Graph::nodes() {
  deque<Graph*> que;
  vector<Node*> ret;
  std::transform(subgraphs_.begin(), subgraphs_.end(), back_inserter(que),
      [](const shared_ptr<Graph>& g)->Graph* {
        return g.get();
      });
  while (que.size() != 0) {
    Graph* g = que.front();
    que.pop_front();
    if (g->subgraphs_.size() == 0) {
      Node* n = dynamic_cast<Node*>(g);
      if (n != NULL && n->rank() == current_rank()) {
        ret.push_back(n);
      }
    } else {
      std::transform(g->subgraphs_.begin(), g->subgraphs_.end(),
          back_inserter(que), [](const shared_ptr<Graph>& g)->Graph* {
            return g.get();
          });
    }
  }
  return ret;
}

/**
 * @brief create blob and add the blob to the graph
 */
Blob* Graph::create(const Size& size, const string& name, int rank,
    int device) {
  subgraphs_.push_back(shared_ptr<Graph>(new Blob(size, rank, device)));
  Graph* g = subgraphs_.rbegin()->get();
  graph_name_[g] = name;
  g->parent_ = this;
  return static_cast<Blob*>(g);
}

Blob* Graph::create(const Size& size, const string& name) {
  return create(size, name, rank_, device_);
}

Blob* Graph::create(shared_ptr<Tensor> tensor, const string& name) {
  subgraphs_.push_back(shared_ptr<Graph>(new Blob(tensor)));
  Graph* g = subgraphs_.rbegin()->get();
  graph_name_[g] = name;
  g->parent_ = this;
  return static_cast<Blob*>(g);
}

}
