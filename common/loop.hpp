#ifndef PURINE_LOOP
#define PURINE_LOOP

#include <thread>
#include <memory>
#include <functional>
#include <mutex>
#include <deque>
#include <atomic>
#include <uv.h>
#include <iostream>

#include "common/common.hpp"
#include "common/cuda.hpp"

using std::atomic;
using std::thread;
using std::mutex;
using std::function;
using std::shared_ptr;
using std::deque;
using namespace std;

namespace purine {

class Loop {
 private:
  Loop(const Loop&);
  Loop& operator=(const Loop&);
 public:
  explicit Loop(int device);
  virtual void post(const function<void()>& fn);
  virtual ~Loop();
 protected:
  int device_;
  mutex mutex_;
  deque<function<void()> > queue_;
  uv_async_t async_;
  uv_loop_t* loop_;
  shared_ptr<thread> thread_;
  atomic<bool> stop_;
  friend void async_cb(uv_async_t* async);
};

}  // namespace purine

#endif
