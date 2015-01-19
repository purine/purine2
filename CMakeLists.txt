cmake_minimum_required (VERSION 2.6)
project (Purine)

# The version number.
set (Purine_VERSION_MAJOR 2)
set (Purine_VERSION_MINOR 0)

#    CMake Scripts dir
set(CMAKE_SCRIPT_DIR ${CMAKE_SOURCE_DIR}/CMakeScripts)

#CMake module path for custom module finding
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SCRIPT_DIR})

find_package (CUDA 6.5 REQUIRED)
include_directories (${CUDA_INCLUDE_DIRS})

# Google-glog
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

# Google-gflags
find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIRS})

# BLAS
find_package(Atlas REQUIRED)
include_directories(${Atlas_INCLUDE_DIR})
set(BLAS_LIBRARIES ${Atlas_LIBRARIES})

find_package(LibUV REQUIRED)
include_directories(${LIBUV_INCLUDE_DIRS})

find_package(MPI)
include_directories(${MPI_INCLUDE_PATH})

include_directories(${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# set_source_files_properties(graph/graph.hpp PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS} -Wno-non-template-friend")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  -gencode arch=compute_20,code=sm_20
  -gencode arch=compute_20,code=sm_21
  -gencode arch=compute_30,code=sm_30
  -gencode arch=compute_35,code=sm_35
  -gencode arch=compute_50,code=sm_50
  -gencode arch=compute_50,code=compute_50
  )

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  -std=c++11
  )

file(GLOB_RECURSE PURINE_CU_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cu)
cuda_add_library(purine_cu STATIC ${PURINE_CU_SOURCES})

file(GLOB_RECURSE PURINE_CPP_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
file(GLOB_RECURSE TEST_CPP_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} test_*.cpp)
list(REMOVE_ITEM PURINE_CPP_SOURCES ${TEST_CPP_SOURCES})

add_library(purine STATIC ${PURINE_CPP_SOURCES})
target_link_libraries(purine purine_cu
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  ${BLAS_LIBRARIES}
  ${MPI_C_LIBRARIES}
  /usr/local/lib/libcudnn.so
  )
CUDA_ADD_CUBLAS_TO_TARGET(purine)
add_subdirectory(tests)
