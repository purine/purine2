// Copyright Lin Min 2014
#include "catch/catch.hpp"
#include "dispatch/runnable.hpp"
#include "composite/layers/concat_layer.hpp"
#include "composite/layers/split_layer.hpp"

using namespace purine;

TEST_CASE("TestConcat", "[Concat]") {
  Runnable run(0, 0);
  Blob* data = run.create("data", { 32, 32, 32, 32 });
  Blob* data_diff = run.create("data_diff", { 32, 32, 32, 32 });
  Blob* data1 = run.create("data1", { 32, 32, 32, 32 });
  Blob* data1_diff = run.create("data1_diff", { 32, 32, 32, 32 });
  Blob* data2 = run.create("data2", { 32, 32, 32, 32 });
  Blob* data2_diff = run.create("data2_diff", { 32, 32, 32, 32 });
  ConcatLayer* concat = run.createGraph<ConcatLayer>("concat",
      ConcatLayer::param_tuple(Split::CHANNELS));
  vector<Blob*>{ data, data1, data2, data_diff, data1_diff, data2_diff }
  >> *concat;
  vector<Blob*> top = concat->top();
  REQUIRE(top[0]->tensor()->size() == Size(32, 96, 32, 32));
  REQUIRE(top[1]->tensor()->size() == Size(32, 96, 32, 32));
  REQUIRE(top[0]->tensor()->stride() == Stride(98304, 1024, 32, 1));
  REQUIRE(top[1]->tensor()->stride() == Stride(98304, 1024, 32, 1));
  REQUIRE(data->tensor()->stride() == Stride(98304, 1024, 32, 1));
  REQUIRE(data1->tensor()->stride() == Stride(98304, 1024, 32, 1));
  REQUIRE(data2->tensor()->stride() == Stride(98304, 1024, 32, 1));
  REQUIRE(data_diff->tensor()->stride() == Stride(98304, 1024, 32, 1));
  REQUIRE(data1_diff->tensor()->stride() == Stride(98304, 1024, 32, 1));
  REQUIRE(data2_diff->tensor()->stride() == Stride(98304, 1024, 32, 1));
  REQUIRE(data->tensor()->gpu_data() == top[0]->tensor()->gpu_data());
  REQUIRE(data1->tensor()->gpu_data() == top[0]->tensor()->gpu_data()
      + 1024 * 32);
  REQUIRE(data2->tensor()->gpu_data() == top[0]->tensor()->gpu_data()
      + 1024 * 64);
  REQUIRE(data_diff->tensor()->gpu_data() == top[1]->tensor()->gpu_data());
  REQUIRE(data1_diff->tensor()->gpu_data() == top[1]->tensor()->gpu_data()
      + 1024 * 32);
  REQUIRE(data2_diff->tensor()->gpu_data() == top[1]->tensor()->gpu_data()
      + 1024 * 64);
}

TEST_CASE("TestSplit", "[Split]") {
  Runnable run(0, 0);
  Blob* data = run.create("data", { 32, 96, 32, 32 });
  Blob* data_diff = run.create("data_diff", { 32, 96, 32, 32 });
  SplitLayer* split = run.createGraph<SplitLayer>("split",
      SplitLayer::param_tuple(Split::CHANNELS, {32, 32, 32}));
  vector<Blob*>{ data, data_diff } >> *split;
  vector<Blob*> out = split->top();
  REQUIRE(data->tensor()->size() == Size(32, 96, 32, 32));
  REQUIRE(data_diff->tensor()->size() == Size(32, 96, 32, 32));
  REQUIRE(data->tensor()->stride() == Stride(98304, 1024, 32, 1));
  REQUIRE(data_diff->tensor()->stride() == Stride(98304, 1024, 32, 1));

  for (int i = 0; i < 6; ++i) {
    REQUIRE(out[i]->tensor()->stride() == Stride(98304, 1024, 32, 1));
  }
  REQUIRE(out[0]->tensor()->gpu_data() == data->tensor()->gpu_data());
  REQUIRE(out[1]->tensor()->gpu_data() == data->tensor()->gpu_data()
      + 1024 * 32);
  REQUIRE(out[2]->tensor()->gpu_data() == data->tensor()->gpu_data()
      + 1024 * 64);
  REQUIRE(out[3]->tensor()->gpu_data() == data_diff->tensor()->gpu_data());
  REQUIRE(out[4]->tensor()->gpu_data() == data_diff->tensor()->gpu_data()
      + 1024 * 32);
  REQUIRE(out[5]->tensor()->gpu_data() == data_diff->tensor()->gpu_data()
      + 1024 * 64);
}
