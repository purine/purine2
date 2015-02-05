// Copyright Lin Min 2015
#include <iostream>
#include "catch/catch.hpp"
#include "operations/operation.hpp"
#include "operations/include/random.hpp"
#include "dispatch/op.hpp"
#include "common/common.hpp"
#include "caffeine/math_functions.hpp"
#include "composite/connectable.hpp"
#include "dispatch/runnable.hpp"
#include "dispatch/graph_template.hpp"
#include "composite/graph/copy.hpp"

using namespace std;
using namespace purine;

typedef vector<Blob*> B;

/**
 * rank:       0           1
 * device: -1, 0, 1    -1, 0, 1
 */
TEST_CASE("TestCopy", "[Copy]") {
  Runnable g;
  SECTION("Single Copy") {
    SECTION("Within rank") {
      int rank = current_rank();
      /**
       * cpu <-> gpu
       * (0, -1) -> (0, 0)
       * (0, 0) -> (0, -1)
       */
      SECTION("cpu <-> gpu") {
        Op<Constant>* c = g.create<Constant>("constant", rank, -1, "main",
            Constant::param_tuple(2.));
        Blob* twos_cpu = g.create("twos_cpu", rank, -1, {1, 10, 10, 10});
        Blob* twos_gpu = g.create("twos_gpu", rank, 0, {1, 10, 10, 10});
        Blob* dest_cpu = g.create("dest_cpu", rank, -1, {1, 10, 10, 10});
        Copy* cp = g.createFlexible<Copy>("cp", Copy::param_tuple());
        Copy* cp2 = g.createFlexible<Copy>("cp2", Copy::param_tuple());

        *c >> B{ twos_cpu } >> *cp >> B{ twos_gpu } >> *cp2 >> B{ dest_cpu };

        REQUIRE(cp->nodes().size() == 1);
        REQUIRE(cp2->nodes().size() == 1);
        REQUIRE(g.nodes().size() == 6);

        g.run();
        REQUIRE(caffe::purine_cpu_compare(twos_cpu->tensor()->cpu_data(),
                dest_cpu->tensor()->cpu_data(),
                twos_cpu->tensor()->size().count()));
      }
      /*
       * gpu <-> gpu
       * (0, 1) -> (0, 0)
       * (0, 0) -> (0, 1)
       */
      SECTION("gpu <-> gpu") {
        Blob* twos_gpu1 = g.create("twos_gpu1", rank, 0, {1, 10, 10, 10});
        Blob* twos_gpu2 = g.create("twos_gpu2", rank, 1, {1, 10, 10, 10});
        Copy* cp = g.createFlexible<Copy>("cp", Copy::param_tuple());
        B{ twos_gpu1 } >> *cp >> B{ twos_gpu2 };

        // filler
        Runnable fill;
        Op<Constant>* c1 = fill.create<Constant>("constant1", rank, 0, "main",
            Constant::param_tuple(2.));
        Op<Constant>* c2 = fill.create<Constant>("constant2", rank, 1, "main",
            Constant::param_tuple(2.));
        Blob* gpu1 = fill.create("gpu1", twos_gpu1->shared_tensor());
        Blob* gpu2 = fill.create("gpu2", twos_gpu2->shared_tensor());
        *c1 >> B{ gpu1 };
        *c2 >> B{ gpu2 };

        REQUIRE(cp->nodes().size() == 1);
        REQUIRE(g.nodes().size() == 3);
        REQUIRE(fill.nodes().size() == 4);

        fill.run();
        REQUIRE(!caffe::purine_gpu_compare(gpu1->tensor()->gpu_data(),
                gpu2->tensor()->gpu_data(),
                gpu1->tensor()->size().count()));
        g.run();
        REQUIRE(caffe::purine_gpu_compare(twos_gpu1->tensor()->gpu_data(),
                twos_gpu2->tensor()->gpu_data(),
                twos_gpu1->tensor()->size().count()));
      }
      /*
       * same device
       * (0, 0) -> (0, 0)
       * (0, -1) -> (0, -1)
       */
      SECTION("gpu <-> gpu") {
        Blob* gpu = g.create("gpu", rank, 0, {1, 10, 10, 10});
        Copy* cp = g.createFlexible<Copy>("cp", Copy::param_tuple(rank, 0));
        B{ gpu } >> *cp;
        cp->top();
        REQUIRE(cp->top() == B{ gpu });
        REQUIRE(g.nodes().size() == 1);
      }
    }

    SECTION("Cross rank") {
      /**
       * cpu <-> cpu
       * (0, -1) <-> (1, -1)
       */
      SECTION("cpu <-> cpu") {
        Blob* rank1 = g.create("rank1", 0, -1, {1, 10, 10, 10});
        Blob* rank2 = g.create("rank2", 1, -1, {1, 10, 10, 10});
        Copy* cp = g.createFlexible<Copy>("cp", Copy::param_tuple());
        B{ rank1 } >> *cp >> B{ rank2 };

        // filler
        Runnable fill;
        Op<Constant>* c1 = fill.create<Constant>("constant1", 0, -1, "main",
            Constant::param_tuple(2.));
        Op<Constant>* c2 = fill.create<Constant>("constant2", 1, -1, "main",
            Constant::param_tuple(1.));
        Op<Constant>* c3 = fill.create<Constant>("constant3", 1, -1, "main",
            Constant::param_tuple(2.));
        Blob* r1 = fill.create("r1", rank1->shared_tensor());
        Blob* r2 = fill.create("r2", rank2->shared_tensor());
        Blob* r3 = fill.create("r3", 1, -1, {1, 10, 10, 10});
        *c1 >> B{ r1 };
        *c2 >> B{ r2 };
        *c3 >> B{ r3 };
        print_graph(g.print());
        print_graph(fill.print());
        // run
        fill.run();
        if (current_rank() == 1) {
          REQUIRE(!caffe::purine_cpu_compare(r2->tensor()->cpu_data(),
                  r3->tensor()->cpu_data(),
                  r2->tensor()->size().count()));
        }
        g.run();
        if (current_rank() == 1) {
          REQUIRE(caffe::purine_cpu_compare(r2->tensor()->cpu_data(),
                  r3->tensor()->cpu_data(),
                  r2->tensor()->size().count()));
        }
      }
      /*
       * gpu -> cpu
       * (0, 0) -> (1, -1)
       */
      /*
       * gpu -> gpu
       * (0, 0) -> (1, 0)
       */
      /*
       * cpu -> gpu
       * (0, -1) -> (0, 0)
       */
    }
  }

  SECTION("MultiCopy") {
    Runnable g;
    Runnable fill;
    vector<Blob*> r2s;
    vector<Blob*> r3s;
    for (int i = 0; i < 10; ++i) {
      Blob* rank1 = g.create("rank1", 0, -1, {1, 10, 10, 10});
      Blob* rank2 = g.create("rank2", 1, -1, {1, 10, 10, 10});
      Copy* cp = g.createFlexible<Copy>("cp", Copy::param_tuple());
      B{ rank1 } >> *cp >> B{ rank2 };

      // filler
      Op<Constant>* c1 = fill.create<Constant>("constant1", 0, -1, "main",
          Constant::param_tuple(DTYPE(i)));
      Op<Constant>* c2 = fill.create<Constant>("constant2", 1, -1, "main",
          Constant::param_tuple(1.414));
      Op<Constant>* c3 = fill.create<Constant>("constant3", 1, -1, "main",
          Constant::param_tuple(DTYPE(i)));
      Blob* r1 = fill.create("r1", rank1->shared_tensor());
      Blob* r2 = fill.create("r2", rank2->shared_tensor());
      Blob* r3 = fill.create("r3", 1, -1, {1, 10, 10, 10});
      *c1 >> B{ r1 };
      *c2 >> B{ r2 };
      *c3 >> B{ r3 };
      r2s.push_back(r2);
      r3s.push_back(r3);
    }
    // run
    fill.run();
    if (current_rank() == 1) {
      for (int i = 0; i < 10; ++i) {
        REQUIRE(!caffe::purine_cpu_compare(r2s[i]->tensor()->cpu_data(),
                r3s[i]->tensor()->cpu_data(),
                r2s[i]->tensor()->size().count()));
      }
    }
    g.run();
    if (current_rank() == 1) {
      for (int i = 0; i < 10; ++i) {
        REQUIRE(caffe::purine_cpu_compare(r2s[i]->tensor()->cpu_data(),
                r3s[i]->tensor()->cpu_data(),
                r2s[i]->tensor()->size().count()));
      }
    }
  }
}
