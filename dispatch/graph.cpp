// Copyright Lin Min 2015
#include <deque>
#include <iterator>
#include "dispatch/graph.hpp"
#include "dispatch/node.hpp"
#include "dispatch/blob.hpp"

using std::deque;

namespace purine {

Graph::Graph(int rank, int device) : rank_(rank), device_(device) {
}

Graph::Graph(const vector<Graph*>& inputs, const vector<Graph*>& outputs,
    int rank, int device) {
}

Graph::~Graph() {
}

/**
 * @fn prepare_once
 * @brief is called only once before first run.
 *        the purpose of this function is to initialize name_ and root_
 *        for all the subgraphs.
 */
void Graph::prepare_once() {
  if (prepared_) {
    return;
  } else {
    prepared_ = true;
    deque<Graph*> que;
    std::transform(subgraphs_.begin(), subgraphs_.end(), back_inserter(que),
        [this](const shared_ptr<Graph>& g)->Graph* {
          g->cached_name_ = "<root>::" + graph_name_[g.get()];
          g->cached_root_ = this;
          return g.get();
        });
    while (que.size() != 0) {
      Graph* front_g = que.front();
      que.pop_front();
      std::transform(front_g->subgraphs_.begin(), front_g->subgraphs_.end(),
          back_inserter(que),
          [this, front_g](const shared_ptr<Graph>& g)->Graph* {
            g->cached_name_ = front_g->cached_name_
                + front_g->graph_name_[g.get()];
            g->cached_root_ = this;
            return g.get();
          });
    }
    cached_sources_ = sources();
    cached_sinks_ = sinks();
  }
}

vector<Node*> Graph::nodes() const {
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
      if (n->rank() == current_rank()) {
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

vector<Node*> Graph::sources() const {
  vector<Node*>&& nodes_ = this->nodes();
  nodes_.erase(std::remove_if(nodes_.begin(), nodes_.end(),
      [](Node* n)->bool { return n->is_source() == false; }), nodes_.end());
  return nodes_;
}

vector<Node*> Graph::sinks() const {
  vector<Node*>&& nodes_ = this->nodes();
  nodes_.erase(std::remove_if(nodes_.begin(), nodes_.end(),
          [](Node* n)->bool { return n->is_sink() == false; }), nodes_.end());
  return nodes_;
}

void Graph::run() {
  run_async();
  sync();
}

void Graph::run_async() {
  prepare_once();
  for (Node* source : cached_sources_) {
    source->run();
  }
}

void Graph::sync() {
  if (sink_counter_ == cached_sinks_.size()) {
    return;
  }
}

void Graph::add_graph(Graph* g) {
}

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

}
