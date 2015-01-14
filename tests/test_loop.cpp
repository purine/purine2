// Copyright Lin Min 2014
#include "catch/catch.hpp"

#include <vector>
#include "common/common.hpp"
#include "common/loop.hpp"

using namespace purine;
using namespace std;

TEST_CASE("TestLoop", "[TestLoop]") {
  int counter = 0;

  SECTION("CPU") {
    Loop* loop = new Loop(-1);
    loop->post([&] () {
          counter++;
        });
    delete loop;
    REQUIRE(1 == counter);
  }

  SECTION("MultiThreading") {
    int num = 10;
    vector<Loop*> loops;
    for (int i = 0; i < num; ++i) {
      Loop* loop = new Loop(-1);
      loops.push_back(loop);
      loop->post([&] () {
            counter++;
          });
    }
    for (int i = 0; i < num; ++i) {
      delete loops[i];
    }
    REQUIRE(num == counter);
  }

  SECTION("GPU") {
    Loop* loop = new Loop(0);
    loop->post([&] () {
          counter++;
        });
    delete loop;
    REQUIRE(1 == counter);
  }

}
