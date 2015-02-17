// Copyright Lin Min 2015
#include <deque>
#include <set>
#include <iterator>
#include <string>
#include <sstream>
#include "dispatch/graph.hpp"
#include "dispatch/node.hpp"
#include "dispatch/blob.hpp"

using std::deque;
using std::stringstream;
using std::set;

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
      if (n != NULL) {
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

string Graph::name() const {
  Graph* p = this->parent_;
  vector<string> name_hirearchy;
  name_hirearchy.push_back(p->graph_name_[this]);
  while (p->parent_ != NULL) {
    name_hirearchy.push_back(p->parent_->graph_name_[p]);
    p = p->parent_;
  }
  stringstream ss;
  while (!name_hirearchy.empty()) {
    ss << name_hirearchy.back();
    name_hirearchy.pop_back();
    if (!name_hirearchy.empty()) {
      ss << "::";
    }
  }
  return ss.str();
}

void Graph::delete_subgraph(Graph* g) {
  // remove the graph
  subgraphs_.erase(std::remove_if(subgraphs_.begin(), subgraphs_.end(),
          [g](const shared_ptr<Graph>& subgraph)->bool {
            return subgraph.get() == g;
          }), subgraphs_.end());
}

/**
 * @brief return nodes that are sources of the graph
 * (which have no preceding nodes);
 */
vector<Node*> Graph::sources() {
  vector<Node*> nodes_ = nodes();
  nodes_.erase(std::remove_if(nodes_.begin(), nodes_.end(),
      [](Node* n)->bool { return n->is_source() == false; }), nodes_.end());
  return nodes_;
}

/**
 * @brief return nodes that are sinks of the graph
 * (which have no postceding nodes);
 */
vector<Node*> Graph::sinks() {
  vector<Node*> nodes_ = nodes();
  nodes_.erase(std::remove_if(nodes_.begin(), nodes_.end(),
          [](Node* n)->bool { return n->is_sink() == false; }), nodes_.end());
  return nodes_;
}

void Graph::prune(const vector<Node*>& to_prune) {
  vector<Node*> sks = this->sinks();
  for (auto sink : to_prune) {
    CHECK(find(sks.begin(), sks.end(), sink) != sks.end());
    sks.erase(remove(sks.begin(), sks.end(), sink), sks.end());
  }
  // traverse graph
  set<Node*> need_;
  deque<Node*> need;
  for (Node* n : sks) {
    need.push_back(n);
  }
  while (!need.empty()) {
    Node* f = need.front();
    need.pop_front();
    if (need_.find(f) == need_.end()) {
      need_.insert(f);
      const vector<Node*>& inputs = f->inputs();
      need.insert(need.end(), inputs.begin(), inputs.end());
    }
  }
  // traverse from not needed sinks
  deque<Node*> no_need(to_prune.begin(), to_prune.end());
  set<Node*> no_need_;
  while (!no_need.empty()) {
    Node* f = no_need.front();
    no_need.pop_front();
    if (need_.find(f) == need_.end() && no_need_.find(f) == no_need_.end()) {
      no_need_.insert(f);
      const vector<Node*>& inputs = f->inputs();
      no_need.insert(no_need.end(), inputs.begin(), inputs.end());
    }
  }
  // LOG
  MPI_LOG( << "Pruning:" );
  for (Node* n : no_need_) {
    MPI_LOG( << n->name() );
    static_cast<Graph*>(n)->parent_->delete_subgraph(n);
  }
}

/**
 * @brief create blob and add the blob to the graph
 */
Blob* Graph::create(const string& name, int rank, int device,
    const Size& size) {
  subgraphs_.push_back(shared_ptr<Graph>(new Blob(rank, device, size)));
  Graph* g = subgraphs_.rbegin()->get();
  graph_name_[g] = name;
  g->parent_ = this;
  return static_cast<Blob*>(g);
}

Blob* Graph::create(const string& name, const Size& size) {
  return create(name, rank_, device_, size);
}

Blob* Graph::create(const string& name, shared_ptr<Tensor> tensor) {
  subgraphs_.push_back(shared_ptr<Graph>(new Blob(tensor)));
  Graph* g = subgraphs_.rbegin()->get();
  graph_name_[g] = name;
  g->parent_ = this;
  return static_cast<Blob*>(g);
}

DTYPE Graph::memory_cost_cpu() {
  DTYPE ret = 0;
  for (auto graph : subgraphs_) {
    if (graph->subgraphs_.empty()) {
      Blob* blob = dynamic_cast<Blob*>(graph.get());
      if (blob && blob->device() < 0) {
        ret += (blob->tensor()->size().count() *
            (sizeof(DTYPE) / 1024. / 1024.));
      }
    } else {
      ret += graph->memory_cost_cpu();
    }
  }
  return ret;
}

DTYPE Graph::memory_cost_gpu() {
  DTYPE ret = 0;
  for (auto graph : subgraphs_) {
    if (graph->subgraphs_.empty()) {
      Blob* blob = dynamic_cast<Blob*>(graph.get());
      if (blob && blob->device() >= 0) {
        ret += (blob->tensor()->size().count() *
            (sizeof(DTYPE) / 1024. / 1024.));
      }
    } else {
      ret += graph->memory_cost_gpu();
    }
  }
  return ret;
}

}
