// Copyright Lin Min 2015

#include "catch/catch.hpp"
#include "dispatch/graph_template.hpp"
#include "dispatch/op_template.hpp"
#include "dispatch/runnable.hpp"
#include "composite/layers/conv_layer.hpp"
#include "composite/layers/inner_prod_layer.hpp"
#include "operations/include/random.hpp"
#include "caffeine/math_functions.hpp"
#include "composite/graph/copy.hpp"
// caffe
#include "caffe/vision_layers.hpp"
#include "caffe/common.hpp"

using namespace purine;
typedef vector<Blob*> B;

void require_near_cpu(const DTYPE* a, const DTYPE* b, int count, DTYPE limit) {
  for (int i = 0; i < count; ++i) {
    INFO("diff: " << a[i] << " " << b[i]);
    REQUIRE(abs(a[i] - b[i]) < limit);
  }
}

void require_near_gpu(const DTYPE* a, const DTYPE* b, int count, DTYPE limit) {
  DTYPE* cpu_a;
  CUDA_CHECK(cudaMallocHost(&cpu_a, sizeof(DTYPE) * count,
          cudaHostAllocPortable));
  DTYPE* cpu_b;
  CUDA_CHECK(cudaMallocHost(&cpu_b, sizeof(DTYPE) * count,
          cudaHostAllocPortable));
  CUDA_CHECK(cudaMemcpy(cpu_a, a, sizeof(DTYPE) * count, cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(cpu_b, b, sizeof(DTYPE) * count, cudaMemcpyDefault));
  for (int i = 0; i < count; ++i) {
    INFO("diff: " << cpu_a[i] << " " << cpu_b[i]);
    REQUIRE(abs(cpu_a[i] - cpu_b[i]) < limit);
  }
  CUDA_CHECK(cudaFreeHost(cpu_a));
  CUDA_CHECK(cudaFreeHost(cpu_b));
}

class PurineConv : public Runnable {
 protected:
  ConvLayer* conv;
  Blob* bottom_;
  Blob* bottom_diff_;
 public:
  PurineConv(int rank, int device, int kernel_size, int num_output, int pad,
      int stride, int group, Size bottom_size) {
    conv = createGraph<ConvLayer>("conv", ConvLayer::param_tuple(pad, pad,
            stride, stride, kernel_size, kernel_size, num_output));
    bottom_ = create(bottom_size, "bottom");
    bottom_diff_ = create(bottom_size, "bottom_diff");
    B{ bottom_, bottom_diff_ } >> *conv;
    conv->top();
  }
  Blob* bottom() { return bottom_; }
  Blob* bottom_diff() { return bottom_diff_; }
  Blob* top() { return conv->top()[0]; }
  Blob* top_diff() { return conv->top()[1]; }
  Blob* weight() { return conv->weight()[0]; }
  Blob* weight_diff() { return conv->weight()[2]; }
  Blob* bias() { return conv->weight()[1]; }
  Blob* bias_diff() { return conv->weight()[3]; }
};


class CaffeConv {
 protected:
  shared_ptr<caffe::Blob<DTYPE> > caffe_bottom;
  shared_ptr<caffe::Blob<DTYPE> > caffe_top;
  shared_ptr<caffe::ConvolutionLayer<DTYPE> > caffe_conv;
 public:
  CaffeConv(int kernel_size, int num_output, int pad, int stride, int group,
      Size bottom_size, bool GPU = true) {
    if (GPU) {
      caffe::Caffe::set_mode(caffe::Caffe::GPU);
    } else {
      caffe::Caffe::set_mode(caffe::Caffe::CPU);
    }
    caffe::LayerParameter layer_param;
    caffe::ConvolutionParameter* convolution_param =
        layer_param.mutable_convolution_param();
    convolution_param->set_kernel_size(kernel_size);
    convolution_param->set_stride(stride);
    convolution_param->set_num_output(num_output);
    convolution_param->set_pad(pad);
    convolution_param->set_group(group);

    caffe_bottom.reset(new caffe::Blob<DTYPE>(bottom_size.num(),
            bottom_size.channels(), bottom_size.height(), bottom_size.width()));
    caffe_top.reset(new caffe::Blob<DTYPE>());

    vector<caffe::Blob<DTYPE>*> bottom_vec = { caffe_bottom.get() };
    vector<caffe::Blob<DTYPE>*> top_vec = { caffe_top.get() };
    caffe_conv.reset(new caffe::ConvolutionLayer<DTYPE>(layer_param));
    caffe_conv->SetUp(bottom_vec, &top_vec);
  }

  caffe::Blob<DTYPE>* bottom() {
    return caffe_bottom.get();
  }

  caffe::Blob<DTYPE>* top() {
    return caffe_top.get();
  }

  caffe::Blob<DTYPE>* weight() {
    return caffe_conv->blobs()[0].get();
  }

  caffe::Blob<DTYPE>* bias() {
    return caffe_conv->blobs()[1].get();
  }

  void run() {
    vector<caffe::Blob<DTYPE>*> bottom_vec = { caffe_bottom.get() };
    vector<caffe::Blob<DTYPE>*> top_vec = { caffe_top.get() };
    caffe_conv->Forward(bottom_vec, &top_vec);
    caffe_conv->Backward(top_vec, { true }, &bottom_vec);
  }
};

TEST_CASE("TestConvolution", "[Convolution]") {

  SECTION("CPU") {
  }

  SECTION("GPU") {
    SECTION("Single Group") {
      PurineConv conv(0, 0, 5, 10, 2, 1, 1, {1, 3, 10, 10});
      *conv.create<Gaussian>(make_tuple(0., 1.), "init", "main")
          >> B{ conv.top_diff(), conv.weight(), conv.bottom(), conv.bias() };
      conv.run();

      CaffeConv caffe_conv(5, 10, 2, 1, 1, {1, 3, 10, 10}, true);
      caffe::caffe_gpu_copy(caffe_conv.bottom()->count(),
          conv.bottom()->tensor()->gpu_data(),
          caffe_conv.bottom()->mutable_gpu_data());
      caffe::caffe_gpu_copy(caffe_conv.top()->count(),
          conv.top_diff()->tensor()->gpu_data(),
          caffe_conv.top()->mutable_gpu_diff());
      caffe::caffe_gpu_copy(caffe_conv.weight()->count(),
          conv.weight()->tensor()->gpu_data(),
          caffe_conv.weight()->mutable_gpu_data());
      caffe::caffe_gpu_copy(caffe_conv.bias()->count(),
          conv.bias()->tensor()->gpu_data(),
          caffe_conv.bias()->mutable_gpu_data());
      caffe_conv.run();

      // check the inputs match
      REQUIRE(caffe::purine_gpu_compare(conv.top_diff()->tensor()->gpu_data(),
              caffe_conv.top()->gpu_diff(), caffe_conv.top()->count()));
      REQUIRE(caffe::purine_gpu_compare(conv.bottom()->tensor()->gpu_data(),
              caffe_conv.bottom()->gpu_data(), caffe_conv.bottom()->count()));
      REQUIRE(caffe::purine_gpu_compare(conv.weight()->tensor()->gpu_data(),
              caffe_conv.weight()->gpu_data(),
              caffe_conv.weight()->count()));
      REQUIRE(caffe::purine_gpu_compare(conv.bias()->tensor()->gpu_data(),
              caffe_conv.bias()->gpu_data(),
              caffe_conv.bias()->count()));
      // check results
      require_near_gpu(conv.top()->tensor()->gpu_data(),
          caffe_conv.top()->gpu_data(),
          caffe_conv.top()->count(), 1e-5);
      require_near_gpu(conv.bottom_diff()->tensor()->gpu_data(),
          caffe_conv.bottom()->gpu_diff(),
          caffe_conv.bottom()->count(), 1e-5);
      require_near_gpu(conv.weight_diff()->tensor()->gpu_data(),
          caffe_conv.weight()->gpu_diff(), caffe_conv.weight()->count(), 1e-5);
      require_near_gpu(conv.bias_diff()->tensor()->gpu_data(),
          caffe_conv.bias()->gpu_diff(), caffe_conv.bias()->count(), 1e-5);
    }

    SECTION("Multi Group") {
      PurineConv conv(0, 0, 5, 10, 2, 1, 1, {1, 3, 10, 10});
      *conv.create<Gaussian>(make_tuple(0., 1.), "init", "main")
          >> B{ conv.top_diff(), conv.weight(), conv.bottom(), conv.bias() };
      conv.run();

      CaffeConv caffe_conv(5, 10, 2, 1, 1, {1, 3, 10, 10}, true);
      caffe::caffe_gpu_copy(caffe_conv.bottom()->count(),
          conv.bottom()->tensor()->gpu_data(),
          caffe_conv.bottom()->mutable_gpu_data());
      caffe::caffe_gpu_copy(caffe_conv.top()->count(),
          conv.top_diff()->tensor()->gpu_data(),
          caffe_conv.top()->mutable_gpu_diff());
      caffe::caffe_gpu_copy(caffe_conv.weight()->count(),
          conv.weight()->tensor()->gpu_data(),
          caffe_conv.weight()->mutable_gpu_data());
      caffe::caffe_gpu_copy(caffe_conv.bias()->count(),
          conv.bias()->tensor()->gpu_data(),
          caffe_conv.bias()->mutable_gpu_data());
      caffe_conv.run();

      // check the inputs match
      REQUIRE(caffe::purine_gpu_compare(conv.top_diff()->tensor()->gpu_data(),
              caffe_conv.top()->gpu_diff(), caffe_conv.top()->count()));
      REQUIRE(caffe::purine_gpu_compare(conv.bottom()->tensor()->gpu_data(),
              caffe_conv.bottom()->gpu_data(), caffe_conv.bottom()->count()));
      REQUIRE(caffe::purine_gpu_compare(conv.weight()->tensor()->gpu_data(),
              caffe_conv.weight()->gpu_data(),
              caffe_conv.weight()->count()));
      REQUIRE(caffe::purine_gpu_compare(conv.bias()->tensor()->gpu_data(),
              caffe_conv.bias()->gpu_data(),
              caffe_conv.bias()->count()));
      // check results
      require_near_gpu(conv.top()->tensor()->gpu_data(),
          caffe_conv.top()->gpu_data(),
          caffe_conv.top()->count(), 1e-5);
      require_near_gpu(conv.bottom_diff()->tensor()->gpu_data(),
          caffe_conv.bottom()->gpu_diff(),
          caffe_conv.bottom()->count(), 1e-5);
      require_near_gpu(conv.weight_diff()->tensor()->gpu_data(),
          caffe_conv.weight()->gpu_diff(), caffe_conv.weight()->count(), 1e-5);
      require_near_gpu(conv.bias_diff()->tensor()->gpu_data(),
          caffe_conv.bias()->gpu_diff(), caffe_conv.bias()->count(), 1e-5);
    }
  }

}



TEST_CASE("TestInnerProduct", "[InnerProduct]") {
  // purine gpu
  Runnable test_inner(0, 0);
  InnerProdLayer* inner = test_inner.createGraph<InnerProdLayer>("inner",
      InnerProdLayer::param_tuple(10));
  Blob* bottom = test_inner.create({1, 3, 10, 10}, "bottom");
  Blob* bottom_diff = test_inner.create({1, 3, 10, 10}, "bottom_diff");
  B{ bottom, bottom_diff } >> *inner;

  Blob* top = inner->top()[0];
  Blob* top_diff = inner->top()[1];
  Blob* weight = inner->weight()[0];
  Blob* weight_diff = inner->weight()[2];
  Blob* bias = inner->weight()[1];
  Blob* bias_diff = inner->weight()[3];

  *test_inner.create<Gaussian>(make_tuple(0., 1.), "init", "main")
      >> B{ top_diff, weight, bottom, bias};
  Blob* top_cpu = test_inner.create(top->tensor()->size(), "cpu_top", 0, -1);
  B{ top } >> *test_inner.createGraph<Copy>("...") >> B{ top_cpu };
  test_inner.run();

  // caffe
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  caffe::LayerParameter layer_param;
  caffe::InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  caffe::Blob<DTYPE> caffe_bottom(1, 3, 10, 10);
  caffe::Blob<DTYPE> caffe_top(1, 10, 1, 1);
  vector<caffe::Blob<DTYPE>*> bottom_vec = { &caffe_bottom };
  vector<caffe::Blob<DTYPE>*> top_vec = { &caffe_top };
  caffe::InnerProductLayer<DTYPE> caffe_inner(layer_param);
  caffe_inner.SetUp(bottom_vec, &top_vec);

  // copy data from purine blob
  caffe::caffe_gpu_copy(caffe_bottom.count(), bottom->tensor()->gpu_data(),
      caffe_bottom.mutable_gpu_data());
  caffe::caffe_gpu_copy(caffe_top.count(), top_diff->tensor()->gpu_data(),
      caffe_top.mutable_gpu_diff());
  caffe::caffe_gpu_copy(caffe_inner.blobs()[0]->count(),
      weight->tensor()->gpu_data(), caffe_inner.blobs()[0]->mutable_gpu_data());
  caffe::caffe_gpu_copy(caffe_inner.blobs()[1]->count(),
      bias->tensor()->gpu_data(), caffe_inner.blobs()[1]->mutable_gpu_data());

  caffe_inner.Forward(bottom_vec, &top_vec);
  caffe_inner.Backward(top_vec, { true }, &bottom_vec);

  require_near_cpu(top_cpu->tensor()->cpu_data(), caffe_top.cpu_data(),
      caffe_top.count(), 0.0001);
}
