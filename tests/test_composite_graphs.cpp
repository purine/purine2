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
        Op<Constant>* c = g.create<Constant>(Constant::param_tuple(2.),
            "constant", rank, -1, "main");
        Blob* twos_cpu = g.create({1, 10, 10, 10}, "twos_cpu", rank, -1);
        Blob* twos_gpu = g.create({1, 10, 10, 10}, "twos_gpu", rank, 0);
        Blob* dest_cpu = g.create({1, 10, 10, 10}, "dest_cpu", rank, -1);
        Copy* cp = g.createGraph<Copy>("cp");
        Copy* cp2 = g.createGraph<Copy>("cp2");

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
        Blob* twos_gpu1 = g.create({1, 10, 10, 10}, "twos_gpu1", rank, 0);
        Blob* twos_gpu2 = g.create({1, 10, 10, 10}, "twos_gpu2", rank, 1);
        Copy* cp = g.createGraph<Copy>("cp");
        B{ twos_gpu1 } >> *cp >> B{ twos_gpu2 };

        // filler
        Runnable fill;
        Op<Constant>* c1 = fill.create<Constant>(Constant::param_tuple(2.),
            "constant1", rank, 0, "main");
        Op<Constant>* c2 = fill.create<Constant>(Constant::param_tuple(1.),
            "constant2", rank, 1, "main");
        Blob* gpu1 = fill.create(twos_gpu1->shared_tensor(), "gpu1");
        Blob* gpu2 = fill.create(twos_gpu2->shared_tensor(), "gpu2");
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
        Blob* gpu = g.create({1, 10, 10, 10}, "gpu", rank, 0);
        Copy* cp = g.createGraph<Copy>("cp", rank, 0);
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
        Blob* rank1 = g.create({1, 10, 10, 10}, "rank1", 0, -1);
        Blob* rank2 = g.create({1, 10, 10, 10}, "rank2", 1, -1);
        Copy* cp = g.createGraph<Copy>("cp");
        B{ rank1 } >> *cp >> B{ rank2 };

        // filler
        Runnable fill;
        Op<Constant>* c1 = fill.create<Constant>(Constant::param_tuple(2.),
            "constant1", 0, -1, "main");
        Op<Constant>* c2 = fill.create<Constant>(Constant::param_tuple(1.),
            "constant2", 1, -1, "main");
        Op<Constant>* c3 = fill.create<Constant>(Constant::param_tuple(2.),
            "constant3", 1, -1, "main");
        Blob* r1 = fill.create(rank1->shared_tensor(), "r1");
        Blob* r2 = fill.create(rank2->shared_tensor(), "r2");
        Blob* r3 = fill.create({1, 10, 10, 10}, "r3", 1, -1);
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
      Blob* rank1 = g.create({1, 10, 10, 10}, "rank1", 0, -1);
      Blob* rank2 = g.create({1, 10, 10, 10}, "rank2", 1, -1);
      Copy* cp = g.createGraph<Copy>("cp");
      B{ rank1 } >> *cp >> B{ rank2 };

      // filler
      Op<Constant>* c1 = fill.create<Constant>(Constant::param_tuple(DTYPE(i)),
          "constant1", 0, -1, "main");
      Op<Constant>* c2 = fill.create<Constant>(Constant::param_tuple(1.414),
          "constant2", 1, -1, "main");
      Op<Constant>* c3 = fill.create<Constant>(Constant::param_tuple(DTYPE(i)),
          "constant3", 1, -1, "main");
      Blob* r1 = fill.create(rank1->shared_tensor(), "r1");
      Blob* r2 = fill.create(rank2->shared_tensor(), "r2");
      Blob* r3 = fill.create({1, 10, 10, 10}, "r3", 1, -1);
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
