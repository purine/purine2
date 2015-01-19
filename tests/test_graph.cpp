// Copyright Lin Min 2015

#include "catch/catch.hpp"
#include "operations/operation.hpp"
#include "operations/include/conv.hpp"
#include "operations/include/random.hpp"
#include "operations/include/eltwise.hpp"
#include "operations/include/copy.hpp"
#include "dispatch/graph_template.hpp"
#include "dispatch/op_template.hpp"
#include "composite/layers/conv_layer.hpp"

using namespace purine;

typedef vector<Blob*> B;

TEST_CASE("TestGraph", "[Graph]") {
  Graph test_graph;
  Op<Conv>* o = test_graph.create<Conv>(Conv::param_tuple(2, 2, 1, 1),
      "conv", "main");
  Blob* bottom = test_graph.create({1, 3, 10, 10}, "bottom", 0, 0);
  Blob* top = test_graph.create({1, 3, 10, 10}, "top", 0, 0);
  Blob* weight = test_graph.create({3, 3, 5, 5}, "weight", 0, 0);
  B{ bottom, weight } >> (*o) >> B{ top };

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

TEST_CASE("RunGraph", "[Graph][Thread]") {
  Graph run_graph;

  /**
   * gaussian_filler >> { bottom, weight } >> conv >> { top }
   */
  SECTION("single convolution") {
    Op<Conv>* o = run_graph.create<Conv>(Conv::param_tuple(2, 2, 1, 1),
        "conv", 0, 0, "main");
    Blob* bottom = run_graph.create({1, 3, 10, 10}, "bottom", 0, 0);
    Blob* top = run_graph.create({1, 4, 10, 10}, "top", 0, 0);
    Blob* weight = run_graph.create({4, 3, 5, 5}, "weight", 0, 0);
    B{ bottom, weight } >> (*o) >> B{ top };

    SECTION("initializer in main thread") {
      Op<Gaussian>* g = run_graph.create<Gaussian>(
          Gaussian::param_tuple(0., 1.), "gaussian", 0, 0, "main");
      (*g) >> B{ bottom, weight };
      run_graph.run();
    }
    SECTION("initializer in another thread") {
      Op<Gaussian>* g = run_graph.create<Gaussian>(
          Gaussian::param_tuple(0., 1.), "gaussian", 0, 0, "main");
      (*g) >> B{ bottom, weight };
      run_graph.run();
    }
  }

  /**
   * constant_filler >> { bottom1, bottom2 } >> sum >> { top }
   *                 >> copy >> { cpu_top }
   */
  SECTION("summation") {
    Op<Sum>* o = run_graph.create<Sum>(Sum::param_tuple(),
        "sum", 0, 0, "main");
    Blob* bottom1 = run_graph.create({1, 3, 10, 10}, "bottom1", 0, 0);
    Blob* bottom2 = run_graph.create({1, 3, 10, 10}, "bottom2", 0, 0);
    Blob* top = run_graph.create({1, 3, 10, 10}, "top", 0, 0);
    Blob* top_cpu = run_graph.create({1, 3, 10, 10}, "top_cpu", 0, -1);
    Op<Copy>* cp = run_graph.create<Copy>(Copy::param_tuple(),
        "cp", 0, 0, "outbound");
    B{ bottom1, bottom2 } >> (*o) >> B{ top };
    B{ top } >> *cp >> B{ top_cpu };
    Op<Constant>* c = run_graph.create<Constant>(Constant::param_tuple(1.),
        "constant", 0, 0, "main");
    (*c) >> B{ bottom1, bottom2 };
    run_graph.run();
    Tensor* t = top_cpu->tensor();
    for (int i = 0; i < t->size().count(); ++i) {
      REQUIRE(t->cpu_data()[i] == 2.);
    }
  }
}

TEST_CASE("Layer", "[Layer][Graph]") {
  Graph g;
  SECTION("ConvLayer") {
    Blob* bottom = g.create({1, 3, 10, 10}, "bottom");
    Blob* top = g.create({1, 4, 10, 10}, "top");
    Blob* weight = g.create({4, 3, 5, 5}, "weight");
    Blob* bias = g.create({1, 4, 1, 1}, "bias");
    Blob* bottom_diff = g.create({1, 3, 10, 10}, "bottom_diff");
    Blob* top_diff = g.create({1, 4, 10, 10}, "top_diff");
    Blob* weight_diff = g.create({4, 3, 5, 5}, "weight_diff");
    Blob* bias_diff = g.create({1, 4, 1, 1}, "bias_diff");
    B bottom_pack = {bottom, bottom_diff};
    B top_pack = {top, top_diff};
    B weight_pack = {weight, bias, weight_diff, bias_diff};

    SECTION("provide nothing") {
      ConvLayer* conv_layer = g.create<ConvLayer>(
          ConvLayer::param_tuple(2, 2, 1, 1, 5, 5, 4), "conv_layer");
      B{ bottom, bottom_diff } >> *conv_layer;
      conv_layer->top();
      REQUIRE(conv_layer->nodes().size() == 11);

      REQUIRE(conv_layer->bottom() == bottom_pack);
      REQUIRE(conv_layer->top() != top_pack);
      REQUIRE(conv_layer->weight() != weight_pack);
    }

    SECTION("provide top") {
      ConvLayer* conv_layer = g.create<ConvLayer>(
          ConvLayer::param_tuple(2, 2, 1, 1, 5, 5, 4), "conv_layer");
      B{ bottom, bottom_diff } >> *conv_layer >> B{ top, top_diff };
      REQUIRE(conv_layer->nodes().size() == 9);
      REQUIRE(conv_layer->bottom() == bottom_pack);
      REQUIRE(conv_layer->top() == top_pack);
      REQUIRE(conv_layer->weight() != weight_pack);
    }

    SECTION("provide weight and top") {
      ConvLayer* conv_layer = g.create<ConvLayer>(
          ConvLayer::param_tuple(2, 2, 1, 1, 5, 5, 4), "conv_layer");
      B{ bottom, bottom_diff } >> *conv_layer >> B{ top, top_diff };
      conv_layer->set_weight({ weight, bias, weight_diff, bias_diff });
      REQUIRE(conv_layer->nodes().size() == 5);
      REQUIRE(conv_layer->bottom() == bottom_pack);
      REQUIRE(conv_layer->top() == top_pack);
      REQUIRE(conv_layer->weight() == weight_pack);
    }
  }
}
