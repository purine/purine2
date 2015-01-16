// Copyright Lin Min 2015
#ifndef PURINE_DISPATCHER
#define PURINE_DISPATCHER

#include <map>

#include "common/loop.hpp"

using std::map;

namespace purine {

class Dispatcher {
 public:
  ~Dispatcher();
  inline static Dispatcher& Get() {
    if (!singleton_.get()) {
      singleton_.reset(new Dispatcher());
    }
    return *singleton_;
  }
  static Loop& task_loop(int device, const string& thread);
 protected:
  static shared_ptr<Dispatcher> singleton_;
  static map<tuple<int, string>, shared_ptr<Loop> > loops_;
 private:
  Dispatcher();
  // disable copy and assignment.
  Dispatcher(const Dispatcher&);
  Dispatcher& operator=(const Dispatcher&);
};

}

#endif
