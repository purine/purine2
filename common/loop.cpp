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

void work_cb(uv_work_t* work) {
  if (((ThreadPool::work_data*)work->data)->fn_) {
    ((ThreadPool::work_data*)work->data)->fn_();
  }
}

void after_work(uv_work_t* work, int status) {
  ThreadPool* l = (ThreadPool*)((ThreadPool::work_data*)work->data)->tp_;
  delete (ThreadPool::work_data*)work->data;
  delete work;
  std::unique_lock<std::mutex> lck(l->mtx_);
  --(l->count_);
  l->cv_.notify_all();
}

void stop_cb(uv_async_t* async) {
  ThreadPool* l = (ThreadPool*)(async->data);
  uv_stop(l->loop_);
  uv_close((uv_handle_t*)&l->async_, NULL);
}

ThreadPool::ThreadPool() {
  loop_ = (uv_loop_t*)malloc(sizeof *loop_);
  UV_CHECK(uv_loop_init(loop_));
  UV_CHECK(uv_async_init(loop_, &async_, stop_cb));
  async_.data = (void*)(this);
  thread_.reset(new thread([=] () {
            UV_CHECK(uv_run(loop_, UV_RUN_DEFAULT));
          }));
}

ThreadPool::~ThreadPool() {
  std::unique_lock<std::mutex> lck(mtx_);
  cv_.wait(lck, [this]()->bool { return count_ == 0; });
  uv_async_send(&async_);
  thread_->join();
  UV_CHECK(uv_loop_close(loop_));
  free(loop_);
}

void ThreadPool::post(const function<void()>& fn) {
  uv_work_t* w = new uv_work_t();
  w->data = new work_data(this, fn);
  std::unique_lock<std::mutex> lck(mtx_);
  ++count_;
  cv_.notify_all();
  uv_queue_work(loop_, w, work_cb, after_work);
}

}
