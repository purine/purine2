# PURINE2 #
purine version 2.

## Directory Structure ##

- #### common ####

  > common codes used across the project. Including abstraction of CUDA,
  abstraction of uv event loop etc.

- #### caffeine ####

  > code taken from Caffe, mainly math functions and some macros from
  common.hpp in Caffe.

- #### catch ####

  > contains the header file of CATCH testing system. It is the unit
  test framework used in Purine. There are not much unit testing in
  this code. Since the core math functions are based on cudnn and
  caffe, it should be no problem. (Though during development I did
  file a bug report to cudnn, now it is fixed in cudnn v2 rc3)

- #### dispatch ####

  > contains definitions of graph, node, op, blob etc.
  blob wraps tensor, op wraps operation. Different from Purine version
  1, there is no standalone dispatcher, the dispatching code is inside blob, op
  and graph. Construction of a graph can be done by connecting blobs
  and ops. The resulting Graph is self-dispatchable. By calling
  graph.run().

- #### composite ####

  > contains predefined composite graphs. which can be used to
  construct larger graphs. For example, all the layers in caffe can be
  defined as a graph in purine. A network can be constructed by
  further connecting these predefined graphs.

- #### operations ####

  > contains operations and tensor. In this version, tensor is 4
  dimensional (It can be changed to ndarray). Operations takes input
  tensors and generate output tensors. Inputs and outputs of a
  operation is stored in a std vector. Operations can take parameters,
  for example, the parameters of convolution contain padding size,
  stride etc. In the operation folder, there are a bunch of predefined
  operations.

- #### tests ####

  > unit tests of the project.

### Tensor and Operation ###

#### Tensor ####

Tensors and operations are the two basic components in purine. Like in
Caffe, tensor in purine is 4 dimensional (num, channel, height, width)
which is convenient for image data. In MPI, `rank` is used to denote
the process id. In Purine `rank` is used for the machine ID (which mean
there is only one process on each machine). A tensor can reside on
different `rank`s, the `rank` of a tensor can be get by calling the
`rank()` function of the tensor. On the same `rank`, tensors can be on
different devices. Thus there is another function `device` which
returns the device id that the tensor resides on. In Purine, negative
device id are reserved for CPU. Id greater than or equal to zero are
for GPUs.

#### Operation ####

The constructor of Operation takes two vector of Tensors. One as input
and one as output. For example convolution operation takes `{ bottom,
weight }` as input and outputs `{ top }`. The constructor of operation
checks that the input and output tensors are correct in size and
location etc. The `compute_cpu` and `compute_gpu` functions are the
code for convolution on cpu and gpu respectively. They takes a `const
vector<bool>&` as argument, which has the same size as `outputs`. This
is to denote whether the computed results should be write to the
output tensor or add to it.
Purine has enough built in operations for daily deep learning usage,
wrapping most of the functions in CUDNN package by NVIDIA.

#### Connection ####

We can almost do everything by defining a bunch of tensors, and
operate on the tensors with different operations sequentially. The
computation logic can be implemented by connecting operations with
tensors, which forms a bi-partite graph (operations never connects
directly to other operations, nor do tensors).

How to execute the calculation sequence stored in the graph?

We want the operations to operate when and only when all its inputs are
ready. We need a counter for the operation, so when each input is
ready it would trigger a `+1` on the counter. When the counter reaches
the number of the inputs, a ready signal is emitted by the operation,
and the operation starts to operate.

The same thing happens to tensor, we want the tensor to emit ready
signal only when results have been received from all the incoming
operations. Thus a counter is also needed for each tensor.

Counter is not part of either operation or tensor, but it is needed
when executing the graph. That's why Op and Blob are introduced here
as wrappers of operation and tensor respectively. So that the counter
can be stored in Op/Blob. In purine, the computation logic is stored
in the bipartite graph consisting of Ops and Blobs.

Connection types:

1. { tensor } >> Op

2. Op >> { tensor }

3. { tensor } >> Connectable

4. Connectable >> { tensor }

5. Connectable >> Connectable

The >> operator works through calling the `set_input` and `set_output`
function in `Connectable`.

Example to construct a graph.

First construct a runnable.

```c++
Runnable run;
```
create nodes in the runnable.

```c++
Blob* bottom = run.create("bottom_name", Size{128, 10, 1, 1});
Blob* weight = run.create("weight_name", Size(16, 10, 1, 1));

Op<Inner>* inner_prod = run.create<Inner>("inner_prod_name", "main", Inner::param_tuple());
// param_tuple is typedefed in the class `Inner`. It is a typedef of tuple<...>.
// It lists the arguments needed when constructing the operation.
// In the case the `Inner` operation does not need any argument.

Blob* top = run.create("top_name", Size(128, 16, 1, 1));
// connect them
vector<Blob*>{ bottom, weight } >> *inner_prod >> vector<Blob*>{ top };

// call run
run.run();
// the graph will be executed (from sources to sinks). But of course you want to set initial values to the Blobs in the real case.
```

### Examples ###
There are two examples under the examples folder.

1. Network in Network on the CIFAR10 dataset which achieves 10.4%
   error rate.

2. GoogLeNet. We run the googlenet on 12 GPUs using data parallelism.
   it is able to converge in 20 hours (If your GPU are highend ones
   which are more stable in temperature, it could be reduced to 17
   hours). The error rate is 12.7% (Higher performance may require
   some tuning as the batch size is quite big as compared to caffe's
   setting.)

Data parallelism is used in both the above examples, because the fully
connected layers are replaced by a global pooling layer, thus the
parameter number is small and suitable for data parallelism.
