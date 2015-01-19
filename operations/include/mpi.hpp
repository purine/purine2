// Copyright Lin Min 2015
#ifndef PURINE_MPI
#define PURINE_MPI

#include <mpi.h>
#include "operations/operation.hpp"
#include "operations/tensor.hpp"

namespace purine {

/**
 * { src } >> isend >> {}
 */
class Isend : public Operation {
 protected:
  int tag;
  int dest;
  MPI_Request request;
 public:
  typedef tuple<int, int> param_tuple;
  explicit Isend(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
      const param_tuple& args);
  virtual void compute_cpu(const vector<bool>& add);
  MPI_Request* mpi_request() { return &request; }
};

/**
 * {} >> irecv >> { dest }
 */
class Irecv : public Operation {
 protected:
  int tag;
  int src;
  MPI_Request request;
 public:
  typedef tuple<int, int> param_tuple;
  explicit Irecv(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
      const param_tuple& args);
  virtual void compute_cpu(const vector<bool>& add);
  MPI_Request* mpi_request() { return &request; }
};

}

#endif
