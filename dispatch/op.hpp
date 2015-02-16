// Copyright Lin Min 2015
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
#include "operations/include/mpi.hpp"
#include "operations/include/mem_copy.hpp"

using std::function;
using std::string;
using std::vector;
using std::shared_ptr;
using std::transform;

namespace purine {

class Blob;

Op_& operator >> (const vector<Blob*>& inputs, Op_& op);
const vector<Blob*>& operator >> (Op_& op, const vector<Blob*>& outputs);

class Op_ : public Node {
  friend class Blob;
 private:
  LoopInterface* loop_ = NULL;
 protected:
  string thread_; // name of thread
  shared_ptr<Operation> o_;
  bool input_setup_ = false;
  bool output_setup_ = false;
 public:
  explicit Op_(int rank, int device, const string& thread);
  virtual ~Op_() override;
  virtual void compute() override;
  inline string thread() const { return thread_; }
  LoopInterface& loop();
  virtual void set_inputs(const vector<Blob*>& inputs);
  virtual void set_outputs(const vector<Blob*>& outputs);
  virtual void check_inputs(const vector<Blob*>& inputs);
  virtual void check_outputs(const vector<Blob*>& outputs);
};

template <typename O>
class Op : public Op_ {
 protected:
  typename O::param_tuple args_;
  virtual void setup() override;
 public:
  explicit Op(int rank, int device, const string& thread,
      const typename O::param_tuple& args);
  inline O* operation() {
    CHECK(o_);
    return dynamic_cast<O*>(o_.get());
  }
};

template <>
class Op<Irecv> : public Op_ {
 protected:
  typename Irecv::param_tuple args_;
  virtual void setup() override;
  // this function tests whether the underlying async mpi operation is done yet
  function<void()> mpi_test_;
 public:
  explicit Op(int rank, int device, const string& thread,
      const typename Irecv::param_tuple& args);
  virtual void compute() override;
};

template <>
class Op<Isend> : public Op_ {
 protected:
  typename Isend::param_tuple args_;
  virtual void setup() override;
  function<void()> mpi_test_;
 public:
  explicit Op(int rank, int device, const string& thread,
      const typename Isend::param_tuple& args);
  virtual void compute() override;
};

template <>
class Op<MemCopy> : public Op_ {
 protected:
  virtual void setup() override;
 public:
  explicit Op(int rank, int device, const string& thread,
      const typename MemCopy::param_tuple& args);
  virtual void set_inputs(const vector<Blob*>& inputs) override;
  virtual void set_outputs(const vector<Blob*>& outputs) override;
  virtual void check_inputs(const vector<Blob*>& inputs) override {}
  virtual void check_outputs(const vector<Blob*>& outputs) override {}
};

}

#endif
