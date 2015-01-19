// Copyright Lin Min 2015
#include "operations/include/mpi.hpp"

namespace purine {

Isend::Isend(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
    const param_tuple& args) : Operation(inputs, outputs) {
  std::tie(tag, dest) = args;
  CHECK_EQ(inputs_.size(), 1);
  CHECK_EQ(outputs_.size(), 0);
}

void Isend::compute_cpu(const vector<bool>& add) {
  MPI_CHECK(MPI_Isend(inputs_[0]->cpu_data(), inputs_[0]->size().count(),
          MPI_FLOAT, dest, tag, MPI_COMM_WORLD, &request));
}

Irecv::Irecv(const vector<Tensor*>& inputs, const vector<Tensor*>& outputs,
    const param_tuple& args) : Operation(inputs, outputs) {
  std::tie(tag, src) = args;
  CHECK_EQ(inputs_.size(), 0);
  CHECK_EQ(outputs_.size(), 1);
}

void Irecv::compute_cpu(const vector<bool>& add) {
  CHECK(!add[0]);
  MPI_CHECK(MPI_Irecv(outputs_[0]->mutable_cpu_data(),
          outputs_[0]->size().count(), MPI_FLOAT, src, tag,
          MPI_COMM_WORLD, &request));
}

}
