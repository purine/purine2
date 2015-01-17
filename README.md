# PURINE2 #
purine version 2.

## Directory Structure ##
- common
  common codes used across the project. Including abstraction of CUDA,
  abstraction of uv event loop etc.

- caffeine
  code taken from Caffe, mainly math functions and some macros from
  common.hpp in Caffe.

- catch
  contains the header file of CATCH testing system. It is the unit
  test framework used in Purine.

- dispatch
  contains definitions of graph, node, op, blob etc.
  blob wraps tensor, op wraps operation. Different from Purine1, there
  is no standalone dispatcher, the dispatching code is inside blob, op
  and graph. Construction of a graph can be done by connecting blobs
  and ops. The resulting Graph is self-dispatchable. By calling
  graph.run().

- graph
  contains predefined graphs. which can be used to construct larger
  graphs. For example, all the layers in caffe can be defined as a
  graph in purine. A network can be constructed by further connecting
  these predefined graphs.

- operations
  contains operations and tensor. In this version, tensor is 4
  dimensional (It can be changed to ndarray). Operations takes input
  tensors and generate output tensors. Inputs and outputs of a
  operation is stored in a std vector. Operations can take parameters,
  for example, the parameters of convolution contain padding size,
  stride etc. In the operation folder, there are a bunch of predefined
  operations.

- tests
  unit tests of the project.
