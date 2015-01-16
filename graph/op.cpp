// Copyright Lin Min 2015
#include <deque>
#include "graph/dispatcher.hpp"
#include "graph/op.hpp"

using std::deque;

namespace purine {

Op_::Op_(int rank, int device, const string& thread)
    : Node(rank, device), thread_(thread) {
}

Op_::~Op_() {
}

// find the loop from dispatcher.
// cache in the loop_ variable, next time would be faster.
Loop& Op_::loop() {
  CHECK_EQ(rank_, current_rank());
  if (loop_ == NULL) {
    loop_ = &Dispatcher::task_loop(device_, thread_);
  }
  return *loop_;
}

}
