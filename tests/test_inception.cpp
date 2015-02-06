// Copyright Lin Min 2015

#include "catch/catch.hpp"
#include "dispatch/graph_template.hpp"
#include "dispatch/op_template.hpp"
#include "dispatch/runnable.hpp"
#include "composite/layers/inception.hpp"
#include "operations/include/random.hpp"
#include "caffeine/math_functions.hpp"
#include "composite/graph/copy.hpp"

TEST_CASE("TestInception", "[Inception]") {
  Runnable run(0, 0);
  Op<Constant>* rnd1 = run.create<Constant>("1", "main",
      Constant::param_tuple(1.));
  Op<Constant>* rnd0 = run.create<Constant>("0", "main",
      Constant::param_tuple(0.));
  Blob* data = run.create("data", {5, 5, 5, 5});
  Blob* data_diff = run.create("data_diff", {5, 5, 5, 5});
  InceptionLayer* inception = run.createGraph<InceptionLayer>("inception",
      InceptionLayer::param_tuple(5, 5, 5, 5, 5, 5));
  vector<Blob*>{ data, data_diff } >> *inception;
  inception->top();
  Copy* cp = run.createFlexible<Copy>("copy", Copy::param_tuple(0, -1));
  vector<Blob*>{ inception->top()[0] } >> *cp;
  // initialization
  vector<Blob*> weight = inception->weight_data();
  *rnd1 >> vector<Blob*>{ data, weight[0], weight[2], weight[4],
        weight[6], weight[8], weight[10] };
  *rnd0 >> vector<Blob*>{ weight[1], weight[3], weight[5],
        weight[7], weight[9], weight[11] };
  cp->top();
  run.run();
  Tensor* out = cp->top()[0]->tensor();
  const DTYPE* dt = out->cpu_data();
  REQUIRE(out->size() == Size(5, 20, 5, 5));
  vector<DTYPE> expected1(25, 5);
  vector<DTYPE> expected2 = { 100, 150, 150, 150, 100,
                              150, 225, 225, 225, 150,
                              150, 225, 225, 225, 150,
                              150, 225, 225, 225, 150,
                              100, 150, 150, 150, 100 };
  vector<DTYPE> expected3 = { 225, 300, 375, 300, 225,
                              300, 400, 500, 400, 300,
                              375, 500, 625, 500, 375,
                              300, 400, 500, 400, 300,
                              225, 300, 375, 300, 225 };
  // check that each sample equivalent.
  for (int i = 0; i < 2500; i += 500) {
    const DTYPE* dti = dt + i;
    for (int j = 0; j < 500; ++j) {
      REQUIRE(dti[j] == dt[j]);
    }
  }
  // check that within a sample first five channel equivalent
  for (int i = 0; i < 500; i += 125) {
    const DTYPE* tmp = dt + i;
    for (int j = 0; j < 125; j += 25) {
      for (int k = 0; k < 25; ++k) {
        REQUIRE(tmp[j + k] == tmp[k]);
      }
    }
  }
  // check that 4 channel blocks satisfy the criteria
  const DTYPE* dt1 = dt + 0;
  const DTYPE* dt2 = dt + 125;
  const DTYPE* dt3 = dt + 250;

  for (int k = 0; k < 25; ++k) {
    cout << " " << to_string(int(dt1[k])) << ((k + 1) % 5 == 0 ? "\n" : "");
  }
  cout << "\n";
  for (int k = 0; k < 25; ++k) {
    REQUIRE(dt1[k] == expected1[k]);
  }
  for (int k = 0; k < 25; ++k) {
    cout << " " << to_string(int(dt2[k])) << ((k + 1) % 5 == 0 ? "\n" : "");
  }
  cout << "\n";
  for (int k = 0; k < 25; ++k) {
    REQUIRE(dt2[k] == expected2[k]);
  }
  for (int k = 0; k < 25; ++k) {
    cout << " " << to_string(int(dt3[k])) << ((k + 1) % 5 == 0 ? "\n" : "");
  }
  for (int k = 0; k < 25; ++k) {
    REQUIRE(dt3[k] == expected3[k]);
  }
}
