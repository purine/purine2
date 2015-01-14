// Copyright Lin Min 2015

#include "catch/catch.hpp"
#include "operations/operation.hpp"
#include "operations/include/conv.hpp"
#include "graph/graph.hpp"

using namespace purine;

class TestGraph : public Graph {
 public:
  TestGraph() {
    Op<Conv>* o = this->create<Conv>(Conv::param_tuple(2, 2, 1, 1),
        "conv", 0, 0);
    Blob* bottom = create({1, 3, 10, 10}, "bottom", 0, 0);
    Blob* top = create({1, 3, 10, 10}, "top", 0, 0);
    Blob* weight = create({3, 3, 5, 5}, "weight", 0, 0);
    vector<Blob*>{ bottom, weight } >> (*o) >> vector<Blob*>{ top };
  }
};

TEST_CASE("TestGraph", "[Graph]") {
  SECTION("construct graph") {
    TestGraph test_graph;
  }
}
