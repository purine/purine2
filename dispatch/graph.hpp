#ifndef PURINE_GRAPH
#define PURINE_GRAPH

#include <algorithm>
#include <atomic>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <condition_variable>

#include "operations/operation.hpp"
#include "operations/tensor.hpp"

namespace purine {

using std::map;
using std::atomic;
using std::string;
using std::vector;
using std::shared_ptr;
using std::mutex;
using std::unique_lock;
using std::condition_variable;

class Node;
template <typename O> class Op;
class Blob;

class SinkCounter {
 private:
  shared_ptr<condition_variable> cv_;
  shared_ptr<mutex> mtx_;
  shared_ptr<int> count_;
 public:
  SinkCounter() {
    cv_.reset(new condition_variable);
    mtx_.reset(new mutex);
    count_.reset(new int(0));
  }
  int operator++ () {
    std::unique_lock<std::mutex> lck(*mtx_);
    ++(*count_);
    cv_->notify_all();
  }
  bool operator== (int num) {
    std::unique_lock<std::mutex> lck(*mtx_);
    cv_->wait(lck, [this, num]()->bool { return *count_ == num; });
    *count_ = 0;
    return true;
  }
};

class Graph {
 protected:
  // All these variables can change when the graph changes.
  // They are initialized by prepare_once() before run.
  bool prepared_ = false;
  void prepare_once();
  string cached_name_;
  Graph* cached_root_;
  vector<Node*> cached_sources_;
  vector<Node*> cached_sinks_;

  int rank_;
  int device_;
  vector<shared_ptr<Graph> > subgraphs_;
  map<Graph*, string> graph_name_;
  Graph* parent_ = NULL;
  SinkCounter sink_counter_;
  virtual void setup() {}
 public:
  explicit Graph(int rank = 0, int device = 0);
  virtual ~Graph();
  virtual void run();
  virtual void run_async();
  virtual void sync();
  virtual vector<Node*> sources() const;
  virtual vector<Node*> sinks() const;
  virtual vector<Node*> nodes() const;

  inline SinkCounter& sink_counter() { return sink_counter_; }

  // create op
  template <typename O>
  Op<O>* create(const typename O::param_tuple& param, const string& name,
      int rank, int device, const string& thread);
  template <typename O>
  Op<O>* create(const typename O::param_tuple& param,
      const string& name, const string& thread);

  // create graph of type G
  template <typename G>
  G* create(const typename G::param_tuple& param, const string& name,
      int rank, int device);
  template <typename G>
  G* create(const typename G::param_tuple& param, const string& name);

  // create blob
  Blob* create(const Size& size, const string& name, int rank, int device);
  Blob* create(const Size& size, const string& name);
  Blob* create(shared_ptr<Tensor> tensor, const string& name);
  inline string cached_name() const { return cached_name_; }
};

}

#endif
