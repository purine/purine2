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

void async_cb(uv_async_t* async) {
  Loop* l = (Loop*)(async->data);
  function<void()> fn;
  while (true) {
    l->mutex_.lock();
    if (l->queue_.size() == 0) {
      if (l->stop_) {
        uv_stop(l->loop_);
        uv_close((uv_handle_t*)&l->async_, NULL);
      }
      l->mutex_.unlock();
      break;
    }
    fn = l->queue_.front();
    l->queue_.pop_front();
    l->mutex_.unlock();
    fn();
  }
}

Loop::Loop(int device) : device_(device), stop_(false) {
  loop_ = (uv_loop_t*)malloc(sizeof *loop_);
  UV_CHECK(uv_loop_init(loop_));
  UV_CHECK(uv_async_init(loop_, &async_, async_cb));
  async_.data = (void*)(this);
  thread_.reset(new thread([=] () {
            if (device_ >= 0) {
              THREAD_SET_CUDA_DEVICE(device);
            }
            UV_CHECK(uv_run(loop_, UV_RUN_DEFAULT));
          }));
}

void Loop::post(const function<void()>& fn) {
  mutex_.lock();
  queue_.push_back(fn);
  mutex_.unlock();
  uv_async_send(&async_);
}

Loop::~Loop() {
  stop_ = true;
  uv_async_send(&async_);
  thread_->join();
  UV_CHECK(uv_loop_close(loop_));
  free(loop_);
}

}  // namespace purine

#endif
