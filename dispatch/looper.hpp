// Copyright Lin Min 2015
#ifndef PURINE_LOOPER
#define PURINE_LOOPER

#include <map>

#include "common/loop.hpp"

using std::map;

namespace purine {

class Looper {
 public:
  ~Looper();
  inline static Looper& Get() {
    if (!singleton_.get()) {
      singleton_.reset(new Looper());
    }
    return *singleton_;
  }
  static Loop& task_loop(int device, const string& thread);
 protected:
  static shared_ptr<Looper> singleton_;
  static map<tuple<int, string>, shared_ptr<Loop> > loops_;
 private:
  Looper();
  // disable copy and assignment.
  Looper(const Looper&);
  Looper& operator=(const Looper&);
};

}

#endif
