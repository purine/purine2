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

class Op_ : public Node {
  friend Op_& operator >> (const vector<Blob*>& inputs, Op_& op);
  friend void operator >> (Op_& op, const vector<Blob*>& outputs);
 private:
  Loop* loop_ = NULL;
 protected:
  string thread_; // name of thread
  shared_ptr<Operation> o_;
 public:
  explicit Op_(int rank, int device, const string& thread);
  virtual ~Op_() override;

  virtual void run() override;

  inline string thread() const { return thread_; }

  Loop& loop();
};

template <typename O>
class Op : public Op_ {
 protected:
  typename O::param_tuple args_;
  virtual void setup() override;
 public:
  explicit Op(const typename O::param_tuple& args,
      int rank, int device, const string& thread);
};

template <>
class Op<Irecv> : public Op_ {
 protected:
  typename Irecv::param_tuple args_;
  virtual void setup() override;
  // this function tests whether the underlying async mpi operation is done yet
  function<void()> mpi_test_;
 public:
  explicit Op(const typename Irecv::param_tuple& args,
      int rank, int device, const string& thread);
  virtual void run() override;
};

template <>
class Op<Isend> : public Op_ {
 protected:
  typename Isend::param_tuple args_;
  virtual void setup() override;
  function<void()> mpi_test_;
 public:
  explicit Op(const typename Isend::param_tuple& args,
      int rank, int device, const string& thread);
  virtual void run() override;
};

template <>
class Op<MemCopy> : public Op_ {
 protected:
  virtual void setup() override;
 public:
  explicit Op(const typename MemCopy::param_tuple& args,
      int rank, int device, const string& thread);
};

}

#endif
