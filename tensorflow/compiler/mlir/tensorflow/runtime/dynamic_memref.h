/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_RUNTIME_DXXX_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_RUNTIME_DXXX_H_

#include <assert.h>
#include <stdlib.h>
#include <vector>
#include <cstdint>

#include "tensorflow/core/framework/tensor.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace runtime {

// Define the struct same to MLIR MemRef descriptor
//
template <int N>
struct MemRefWrapper {
  void* allocated_ptr;
  void* aligned_ptr;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

namespace {

template <int N>
MemRefWrapper<N>* AllocateMemrefDescriptor(int64_t rank) {
  auto* descriptor =
      reinterpret_cast<MemRefWrapper<N>*>(malloc(sizeof(MemRefWrapper<N>)));
  assert(descriptor != nullptr &&
         "Malloc MemRefWrapper failed.");

  descriptor->allocated_ptr = nullptr;
  descriptor->aligned_ptr = nullptr;
  descriptor->offset = 0;

  return descriptor;
}

template <int N>
std::vector<void*> AllocateMemrefArgs(std::vector<int64_t>& ranks) {
  std::vector<void*> args;
  for (auto rank : ranks) {
    auto descriptor = AllocateMemrefDescriptor<N>(rank);
    args.push_back(descriptor);
  }

  return args;
}

template <int N>
void* AllocateMemrefArg(int64_t rank) {
  return AllocateMemrefDescriptor<N>(rank);
}

std::vector<int64_t> MakeStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> tmp;
  if (shape.empty()) {
    return tmp;
  }
  tmp.reserve(shape.size());
  int64_t cur_stride = 1;
  for (auto rit = shape.rbegin(), reit = shape.rend(); rit != reit; ++rit) {
    assert(*rit > 0 &&
           "size must be greater than 0 along all dimensions of shape");
    tmp.push_back(cur_stride);
    cur_stride *= *rit;
  }

  return std::vector<int64_t>(tmp.rbegin(), tmp.rend());
}

}

template <int N>
void* BuildMemrefArgument(tensorflow::Tensor& arg) {
  int64_t rank = arg.shape().dims();
  auto void_arg = AllocateMemrefArg<N>(rank);
  auto memref_arg = reinterpret_cast<MemRefWrapper<N>*>(void_arg);
  void* tmp_tensor_data = const_cast<void*>(static_cast<const void*>(arg.tensor_data().data()));

  memref_arg->allocated_ptr = tmp_tensor_data;
  memref_arg->aligned_ptr = tmp_tensor_data;
  std::vector<int64_t> shape;
  for (auto i = 0; i < rank; ++i) {
    shape.push_back(arg.shape().dim_sizes()[i]);
  }
  std::vector<int64_t> stride = MakeStrides(shape);

  for (auto i = 0; i < rank; ++i) {
    memref_arg->sizes[i] = shape[i];
    memref_arg->strides[i] = stride[i];
  }

  MemRefWrapper<N>** ret_arg = reinterpret_cast<MemRefWrapper<N>**>(malloc(sizeof(MemRefWrapper<N>*)));
  *ret_arg = memref_arg;

  return (void*)(ret_arg);
}

template <int N>
void* BuildMemrefArgument(void* ptr, const std::vector<int64_t>& shape) {
  int64_t rank = shape.size();
  auto void_arg = AllocateMemrefArg<N>(rank);
  auto memref_arg = reinterpret_cast<MemRefWrapper<N>*>(void_arg);
  memref_arg->allocated_ptr = ptr;
  memref_arg->aligned_ptr = ptr;
  std::vector<int64_t> stride = MakeStrides(shape);
  for (auto i = 0; i < rank; ++i) {
    memref_arg->sizes[i] = shape[i];
    memref_arg->strides[i] = stride[i];
  }

  MemRefWrapper<N>** ret_arg = reinterpret_cast<MemRefWrapper<N>**>(malloc(sizeof(MemRefWrapper<N>*)));
  *ret_arg = memref_arg;

  return (void*)(ret_arg);
}

} // namespace mlir
} // namespace mlir_runtime

#endif 
