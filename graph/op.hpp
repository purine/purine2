#ifndef PURINE_OP
#define PURINE_OP

#include <algorithm>
#include <functional>
#include <string>
#include <vector>
#include <memory>

#include "common/loop.hpp"
#include "graph/blob.hpp"
#include "graph/node.hpp"
#include "operations/operation.hpp"
#include "operations/tensor.hpp"

using std::function;
using std::string;
using std::vector;
using std::shared_ptr;
using std::transform;

namespace purine {

class Blob;

class Op_ : public Node {
 private:
  Loop* loop_ = NULL;
 protected:
  string thread_;
  function<void()> fn_;
  shared_ptr<Operation> o_;
 public:
  explicit Op_(int rank, int device, const string& thread);
  virtual ~Op_();

  inline string thread() const { return thread_; }

  Loop& loop();
};

template <typename O>
class Op : public Op_ {
  template <typename U>
  friend Op<U>& operator >> (const vector<Blob*>& inputs, Op<U>& op);
  template <typename U>
  friend void operator >> (Op<U>& op, const vector<Blob*>& outputs);
 protected:
  typename O::param_tuple args_;
 public:
  explicit Op(const typename O::param_tuple& args,
      int rank, int device, const string& thread)
      : Op_(rank, device, thread), args_(args) {
  }
  explicit Op(const typename O::param_tuple& args,
      const initializer_list<Blob*>& inputs,
      const initializer_list<Blob*>& outputs,
      int rank, int device, const string& thread)
      : Op(args, rank, device, thread) {
    inputs_ = inputs;
    outputs_ = outputs;
  }
  virtual void run();
};

}

#endif
