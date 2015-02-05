// Copyright Lin Min 2014
#include "catch/catch.hpp"
#include <vector>
#include "operations/include/random.hpp"
#include "operations/operation.hpp"
#include "operations/include/mpi.hpp"
#include "dispatch/op_template.hpp"
#include "dispatch/graph_template.hpp"
#include "dispatch/runnable.hpp"
#include "dispatch/blob.hpp"

using namespace purine;
using namespace std;

typedef vector<Blob*> B;

TEST_CASE("TestMPI", "[MPI][Thread]") {
  Runnable g;
  SECTION("Isend and Irecv") {
    int rank = current_rank();
    Blob* src = g.create("src", 0, -1, {1, 3, 10, 10});
    Blob* dest = g.create("dest", 1, -1, {1, 3, 10, 10});
    Op<Constant>* c = g.create<Constant>("constant", 0, -1, "main",
        Constant::param_tuple(1.));
    Op<Isend>* send = g.create<Isend>("send", 0, -1, "main",
        Isend::param_tuple(0, dest->rank()));
    Op<Irecv>* recv = g.create<Irecv>("recv", 1, -1, "main",
        Irecv::param_tuple(0, src->rank()));
    // connect
    *c >> B{ src };
    B{ src } >> *send;
    *recv >> B{ dest };
    // run
    g.run();
    Tensor* t = dest->tensor();
    if (current_rank() == 1) {
      for (int i = 0; i < t->size().count(); ++i) {
        REQUIRE(t->cpu_data()[i] == 1.);
      }
    }
  }
}
