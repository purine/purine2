// Copyright Lin Min 2015
#include "common/loop.hpp"
namespace purine {
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
}
