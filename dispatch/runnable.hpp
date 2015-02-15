// Copyright Lin Min 2015
#ifndef PURINE_RUNNABLE
#define PURINE_RUNNABLE

#include <condition_variable>
#include <map>
#include <mutex>

#include "common/loop.hpp"
#include "dispatch/graph.hpp"
#include "dispatch/node.hpp"

using std::mutex;
using std::unique_lock;
using std::condition_variable;

namespace purine {

class Runnable : public Graph {
 public:

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

 protected:
  vector<Node*> cached_sources_;
  vector<Node*> cached_sinks_;
  bool prepared_ = false;
  void prepare_once();
  SinkCounter sink_counter_;
  mutex mutex_;
  map<tuple<int, string>, shared_ptr<Loop> > loops_;
 public:
  explicit Runnable(int rank = 0, int device = 0);
  virtual ~Runnable();

  inline SinkCounter& sink_counter() { return sink_counter_; }

  Loop& task_loop(int device, const string& thread);
  virtual void run();
  virtual void run_async();
  virtual void sync();
  vector<Node*> sources();
  vector<Node*> sinks();
  vector<vector<string> > print();
};

}

#endif
