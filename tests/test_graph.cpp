// Copyright Lin Min 2015

#include "catch/catch.hpp"
#include "operations/operation.hpp"
#include "operations/include/conv.hpp"
#include "operations/include/random.hpp"
#include "dispatch/graph_template.hpp"
#include "dispatch/op_template.hpp"

using namespace purine;

class TestGraph : public Graph {
 public:
  typedef vector<Blob*> B;
  TestGraph() {
    Op<Conv>* o = this->create<Conv>(Conv::param_tuple(2, 2, 1, 1),
        "conv", "main");
    Blob* bottom = create({1, 3, 10, 10}, "bottom", 0, 0);
    Blob* top = create({1, 3, 10, 10}, "top", 0, 0);
    Blob* weight = create({3, 3, 5, 5}, "weight", 0, 0);
    B{ bottom, weight } >> (*o) >> B{ top };
  }
};

TEST_CASE("TestGraph", "[Graph]") {
  TestGraph test_graph;
  SECTION("graph node number test") {
    vector<Node*> nodes = test_graph.nodes();
    int node_size = test_graph.nodes().size();
    int source_size = test_graph.sources().size();
    int sink_size = test_graph.sinks().size();
    REQUIRE(node_size == 4);
    REQUIRE(source_size == 2);
    REQUIRE(sink_size == 1);
  }
  SECTION("graph sources and sinks test") {
    vector<Node*>&& nodes = test_graph.nodes();
    // test sources
    REQUIRE(nodes[0]->is_source() == false);
    REQUIRE(nodes[1]->is_source() == true);
    REQUIRE(nodes[2]->is_source() == false);
    REQUIRE(nodes[3]->is_source() == true);
    // test sinks
    REQUIRE(nodes[0]->is_sink() == false);
    REQUIRE(nodes[1]->is_sink() == false);
    REQUIRE(nodes[2]->is_sink() == true);
    REQUIRE(nodes[3]->is_sink() == false);

  }
}

class RunGraph : public Graph {
 public:
  typedef vector<Blob*> B;
  RunGraph() {
    Op<Conv>* o = this->create<Conv>(Conv::param_tuple(2, 2, 1, 1),
        "conv", 0, 0, "main");
    Blob* bottom = create({1, 3, 10, 10}, "bottom", 0, 0);
    Blob* top = create({1, 4, 10, 10}, "top", 0, 0);
    Blob* weight = create({4, 3, 5, 5}, "weight", 0, 0);
    B{ bottom, weight } >> (*o) >> B{ top };

    // initializer
    Op<Gaussian>* g = this->create<Gaussian>(Gaussian::param_tuple(0., 1.),
        "gaussian", 0, 0, "main");
    (*g) >> B{ bottom, weight };
  }
};

TEST_CASE("RunGraph", "[Graph][Thread]") {
  RunGraph run_graph;
  SECTION("run this little graph") {
    run_graph.run();
  }
}
