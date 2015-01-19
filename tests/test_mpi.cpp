// Copyright Lin Min 2014
#include "catch/catch.hpp"
#include <vector>
#include "operations/include/random.hpp"
#include "operations/operation.hpp"
#include "operations/include/mpi.hpp"
#include "dispatch/op_template.hpp"
#include "dispatch/graph_template.hpp"
#include "dispatch/blob.hpp"

using namespace purine;
using namespace std;

typedef vector<Blob*> B;

TEST_CASE("TestMPI", "[MPI][Thread]") {
  Graph g;
  SECTION("Isend and Irecv") {
    int rank = current_rank();
    Blob* src = g.create({1, 3, 10, 10}, "src", 0, -1);
    Blob* dest = g.create({1, 3, 10, 10}, "dest", 1, -1);
    Op<Constant>* c = g.create<Constant>(Constant::param_tuple(1.),
        "constant", 0, -1, "main");
    Op<Isend>* send = g.create<Isend>(Isend::param_tuple(0, 1), "send", 0, -1,
        "main");
    Op<Irecv>* recv = g.create<Irecv>(Irecv::param_tuple(0, 0), "recv", 1, -1,
        "main");
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
