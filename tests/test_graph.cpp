// Copyright Lin Min 2015

#include "catch/catch.hpp"
#include "operations/operation.hpp"
#include "operations/include/conv.hpp"
#include "operations/include/random.hpp"
#include "operations/include/eltwise.hpp"
#include "operations/include/mem_copy.hpp"
#include "dispatch/graph_template.hpp"
#include "dispatch/op_template.hpp"
#include "dispatch/runnable.hpp"
#include "composite/layers/conv_layer.hpp"

using namespace purine;

typedef vector<Blob*> B;

TEST_CASE("TestGraph", "[Graph]") {
  Runnable test_graph;
  Op<Conv>* o = test_graph.create<Conv>("conv", "main",
      Conv::param_tuple(2, 2, 1, 1));
  Blob* bottom = test_graph.create("bottom", 0, 0, {1, 3, 10, 10});
  Blob* top = test_graph.create("top", 0, 0, {1, 3, 10, 10});
  Blob* weight = test_graph.create("weight", 0, 0, {3, 3, 5, 5});
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
  Runnable run_graph;

  /**
   * gaussian_filler >> { bottom, weight } >> conv >> { top }
   */
  SECTION("single convolution") {
    Op<Conv>* o = run_graph.create<Conv>("conv", 0, 0, "main",
        Conv::param_tuple(2, 2, 1, 1));
    Blob* bottom = run_graph.create("bottom", 0, 0, {1, 3, 10, 10});
    Blob* top = run_graph.create("top", 0, 0, {1, 4, 10, 10});
    Blob* weight = run_graph.create("weight", 0, 0, {4, 3, 5, 5});
    B{ bottom, weight } >> (*o) >> B{ top };

    SECTION("initializer in main thread") {
      Op<Gaussian>* g = run_graph.create<Gaussian>("gaussian", 0, 0, "main",
          Gaussian::param_tuple(0., 1.));
      (*g) >> B{ bottom, weight };
      run_graph.run();
    }
    SECTION("initializer in another thread") {
      Op<Gaussian>* g = run_graph.create<Gaussian>("gaussian", 0, 0, "main",
          Gaussian::param_tuple(0., 1.));
      (*g) >> B{ bottom, weight };
      run_graph.run();
    }
  }

  /**
   * constant_filler >> { bottom1, bottom2 } >> sum >> { top }
   *                 >> copy >> { cpu_top }
   */
  SECTION("summation") {
    Op<Sum>* o = run_graph.create<Sum>("sum", 0, 0, "main", Sum::param_tuple());
    Blob* bottom1 = run_graph.create("bottom1", 0, 0, {1, 3, 10, 10});
    Blob* bottom2 = run_graph.create("bottom2", 0, 0, {1, 3, 10, 10});
    Blob* top = run_graph.create("top", 0, 0, {1, 3, 10, 10});
    Blob* top_cpu = run_graph.create("top_cpu", 0, -1, {1, 3, 10, 10});
    Op<MemCopy>* cp = run_graph.create<MemCopy>("cp", "outbound",
        MemCopy::param_tuple());
    B{ bottom1, bottom2 } >> (*o) >> B{ top };
    B{ top } >> *cp >> B{ top_cpu };
    Op<Constant>* c = run_graph.create<Constant>("constant", 0, 0, "main",
        Constant::param_tuple(1.));
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
    Blob* bottom = g.create("bottom", {1, 3, 10, 10});
    Blob* top = g.create("top", {1, 4, 10, 10});
    Blob* weight = g.create("weight", {4, 3, 5, 5});
    Blob* bias = g.create("bias", {1, 4, 1, 1});
    Blob* bottom_diff = g.create("bottom_diff", {1, 3, 10, 10});
    Blob* top_diff = g.create("top_diff", {1, 4, 10, 10});
    Blob* weight_diff = g.create("weight_diff", {4, 3, 5, 5});
    Blob* bias_diff = g.create("bias_diff", {1, 4, 1, 1});
    B bottom_pack = {bottom, bottom_diff};
    B top_pack = {top, top_diff};
    B weight_pack = {weight, bias, weight_diff, bias_diff};

    SECTION("provide nothing") {
      ConvLayer* conv_layer = g.createGraph<ConvLayer>("conv_layer",
          ConvLayer::param_tuple(2, 2, 1, 1, 5, 5, 4, ""));
      B{ bottom, bottom_diff } >> *conv_layer;
      conv_layer->top();
      REQUIRE(conv_layer->bottom() == bottom_pack);
      REQUIRE(conv_layer->top() != top_pack);
      REQUIRE(conv_layer->weight() != weight_pack);
    }

    SECTION("provide top") {
      ConvLayer* conv_layer = g.createGraph<ConvLayer>("conv_layer",
          ConvLayer::param_tuple(2, 2, 1, 1, 5, 5, 4, ""));
      B{ bottom, bottom_diff } >> *conv_layer >> B{ top, top_diff };
      REQUIRE(conv_layer->bottom() == bottom_pack);
      REQUIRE(conv_layer->top() == top_pack);
      REQUIRE(conv_layer->weight() != weight_pack);
    }

    SECTION("provide weight and top") {
      ConvLayer* conv_layer = g.createGraph<ConvLayer>("conv_layer",
          ConvLayer::param_tuple(2, 2, 1, 1, 5, 5, 4, ""),
          B{ weight, bias, weight_diff, bias_diff });
      B{ bottom, bottom_diff } >> *conv_layer >> B{ top, top_diff };
      REQUIRE(conv_layer->bottom() == bottom_pack);
      REQUIRE(conv_layer->top() == top_pack);
      REQUIRE(conv_layer->weight() == weight_pack);
    }
  }
}

TEST_CASE("Operator", "[Layer][Graph]") {
  Runnable g;
  Blob* bottom = g.create("bottom", {1, 3, 10, 10});
  Blob* bottom_diff = g.create("bottom_diff", {1, 3, 10, 10});
  B bottom_pack = {bottom, bottom_diff};
  // convolution layer with bottom
  ConvLayer* conv_layer = g.createGraph<ConvLayer>("conv_layer",
      ConvLayer::param_tuple(2, 2, 1, 1, 5, 5, 4, ""));
  ConvLayer* conv_layer2 = g.createGraph<ConvLayer>("conv_layer2", 1, 1,
      ConvLayer::param_tuple(2, 2, 1, 1, 5, 5, 4, ""));
  bottom_pack >> *conv_layer >> *conv_layer2;
  conv_layer2->top();
  print_graph(g.print());
}
