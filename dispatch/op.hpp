#ifndef PURINE_OP
#define PURINE_OP

#include <algorithm>
#include <functional>
#include <string>
#include <vector>
#include <memory>

#include "common/loop.hpp"
#include "dispatch/blob.hpp"
#include "dispatch/node.hpp"
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
  friend Op_& operator >> (const vector<Blob*>& inputs, Op_& op);
  friend void operator >> (Op_& op, const vector<Blob*>& outputs);
 private:
  Loop* loop_ = NULL;
 protected:
  string thread_;
  function<void()> fn_;
  shared_ptr<Operation> o_;
 public:
  explicit Op_(int rank, int device, const string& thread);
  virtual ~Op_();
  virtual void run();
  virtual void setup() = 0;
  inline string thread() const { return thread_; }

  Loop& loop();
};

template <typename O>
class Op : public Op_ {
 protected:
  typename O::param_tuple args_;
 public:
  explicit Op(const typename O::param_tuple& args,
      int rank, int device, const string& thread);
  explicit Op(const typename O::param_tuple& args,
      const initializer_list<Blob*>& inputs,
      const initializer_list<Blob*>& outputs,
      int rank, int device, const string& thread);
  virtual void setup();
};

}

#endif
