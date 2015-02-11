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
#include "composite/graph/copy.hpp"

using namespace purine;
using namespace std;

typedef vector<Blob*> B;

TEST_CASE("TestDistribute", "[Distribute]") {
  Runnable g;
  Blob*source = g.create("source", 0, -1, {10, 10, 10, 10});
  Blob* dest1 = g.create("dest1", 0, 0, {10, 10, 10, 10});
  Blob* dest2 = g.create("dest2", 0, 1, {10, 10, 10, 10});
  Blob* dest3 = g.create("dest3", 0, 2, {10, 10, 10, 10});
  B{ source } >> *g.createAny<Distribute>("dist",
      Distribute::param_tuple()) >> B{ dest1, dest2, dest3 };
  print_graph(g.print());
}

TEST_CASE("TestAggregate", "[Aggregate]") {
  Runnable g;
  Op<Constant>* constant1 = g.create<Constant>("constant", 0, 0, "main",
      Constant::param_tuple(1.));
  Op<Constant>* constant2 = g.create<Constant>("constant", 0, 1, "main",
      Constant::param_tuple(3.));
  Op<Constant>* constant3 = g.create<Constant>("constant", 0, 2, "main",
      Constant::param_tuple(5.));
  Blob* dest = g.create("dest", 0, -1, {10, 10, 10, 10});
  Blob* src1 = g.create("src1", 0, 0, {10, 10, 10, 10});
  Blob* src2 = g.create("src2", 0, 1, {10, 10, 10, 10});
  Blob* src3 = g.create("src3", 0, 2, {10, 10, 10, 10});
  *constant1 >> B{ src1 };
  *constant2 >> B{ src2 };
  *constant3 >> B{ src3 };
  B{ src1, src2, src3 } >> *g.createAny<Aggregate>("agg",
      Aggregate::param_tuple(Aggregate::AVERAGE, 0, 0)) >> B{ dest };
  print_graph(g.print());
  g.run();
  INFO("dest[0]: " << dest->tensor()->cpu_data()[0]);
  for (int i = 0; i < 10000; ++i) {
    REQUIRE(dest->tensor()->cpu_data()[i] == 3.);
  }
}
