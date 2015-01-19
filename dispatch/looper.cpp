// Copyright Lin Min 2015
#include "dispatch/looper.hpp"

namespace purine {

map<tuple<int, string>, shared_ptr<Loop> > Looper::loops_;
shared_ptr<Looper> Looper::singleton_;
mutex Looper::mutex_;

Looper::Looper() {
}

Loop& Looper::task_loop(int device, const string& thread) {
  mutex_.lock();
  tuple<int, string> key = make_tuple(device, thread);
  if (loops_.count(key) == 0) {
    loops_[key] = shared_ptr<Loop>(new Loop(device));
  }
  Loop& ret = *loops_[key];
  mutex_.unlock();
  return ret;
}

}
