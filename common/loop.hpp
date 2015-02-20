#ifndef PURINE_LOOP
#define PURINE_LOOP

#include <thread>
#include <memory>
#include <functional>
#include <condition_variable>
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

class LoopInterface {
 public:
  virtual void post(const function<void()>& fn) = 0;
};

class Loop : public LoopInterface {
 private:
  Loop(const Loop&);
  Loop& operator=(const Loop&);
 public:
  explicit Loop(int device);
  virtual void post(const function<void()>& fn);
  virtual ~Loop();
 protected:
  Loop() : stop_(false) {}
  int device_;
  mutex mutex_;
  deque<function<void()> > queue_;
  uv_async_t async_;
  uv_loop_t* loop_;
  shared_ptr<thread> thread_;
  atomic<bool> stop_;
  friend void async_cb(uv_async_t* async);
};

class ThreadPool : public Loop {
 private:
  ThreadPool(const ThreadPool&);
  ThreadPool& operator=(const ThreadPool&);
 public:
  explicit ThreadPool();
  virtual ~ThreadPool() {}
 protected:
  struct work_data {
    work_data(ThreadPool* tp, const function<void()>& fn) : tp_(tp), fn_(fn) {}
    ThreadPool* tp_;
    function<void()> fn_;
  };
  int count_ = 0;
  friend void work_cb(uv_work_t* work);
  friend void after_work(uv_work_t* work, int status);
  friend void thread_pool_async_cb(uv_async_t* async);
};

}  // namespace purine

#endif
