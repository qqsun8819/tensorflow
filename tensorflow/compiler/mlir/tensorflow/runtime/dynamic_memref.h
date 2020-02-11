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

typedef enum {
  I32Type = 0,
  I64Type = 1,
  F32Type = 2,
  F64Type = 3,

  InvalidType = 100,
} ElementType;

inline ElementType GetElementType(const std::string& type) {
  if (type == "i32") {
    return ElementType::I32Type;
  } else if (type == "i64") {
    return ElementType::I64Type;
  } else if (type == "f32") {
    return ElementType::F32Type;
  } else if (type == "f64") {
    return ElementType::F64Type;
  } else {
    assert(false && "Not support enum ElementType now.");
    return ElementType::InvalidType;
  }
}

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

// For get result tensor from compiled function
struct ResultTensorWrapper {
  ResultTensorWrapper() {
    result_tensor_addr_saver_ = (int64_t*)malloc(sizeof(int64_t));
    *result_tensor_addr_saver_ = 0;
    result_tensor_addr2_saver_ = (int64_t**)malloc(sizeof(int64_t*));
    *result_tensor_addr2_saver_ = result_tensor_addr_saver_;
    int64_t result_addr2_i64 = (int64_t)(result_tensor_addr2_saver_);
    result_addr2_i64_pointer_ = (int64_t*)malloc(sizeof(int64_t));
    *result_addr2_i64_pointer_ = result_addr2_i64;
  }

  ~ResultTensorWrapper() {
    // NOTE(jiankeng.pt):
    // delete tensor, tensor should be copied before ~ResultTensorWrapper() be called
    tensorflow::Tensor* t = (tensorflow::Tensor*)(*result_tensor_addr_saver_);
    delete t;

    delete result_tensor_addr_saver_;
    delete result_tensor_addr2_saver_;
    delete result_addr2_i64_pointer_;
    result_tensor_addr_saver_ = nullptr;
    result_tensor_addr2_saver_ = nullptr;
    result_addr2_i64_pointer_ = nullptr;
  }

  void* GetArg() {
    return (void*)(result_addr2_i64_pointer_);
  }

  tensorflow::Tensor* GetResultTensorPointer() {
    if (*result_tensor_addr_saver_ == 0) {
      LOG(FATAL) << "Get result tensor from compiled function failed.";
    }
    return (tensorflow::Tensor*)(*result_tensor_addr_saver_);
  }

  int64_t* result_tensor_addr_saver_;
  int64_t** result_tensor_addr2_saver_;
  int64_t* result_addr2_i64_pointer_;
};

// For input args which are passed to compiled function
template <int N>
struct InputTensorWrapper {
  InputTensorWrapper(tensorflow::Tensor& arg) {
    int64_t rank = arg.shape().dims();
    if (rank != N) {
      LOG(FATAL) << "Input tensor's shape vs input shape N, " << rank << " : " << N;
    }
    memref_wrapper_pointer_ = AllocateMemrefDescriptor<N>(rank);
    void* tmp_tensor_data = const_cast<void*>(static_cast<const void*>(arg.tensor_data().data()));
    memref_wrapper_pointer_->allocated_ptr = tmp_tensor_data;
    memref_wrapper_pointer_->aligned_ptr = tmp_tensor_data;
    std::vector<int64_t> shape;
    for (auto i = 0; i < rank; ++i) {
      shape.push_back(arg.shape().dim_sizes()[i]);
    }

    std::vector<int64_t> stride = MakeStrides(shape);
    for (auto i = 0; i < rank; ++i) {
      memref_wrapper_pointer_->sizes[i] = shape[i];
      memref_wrapper_pointer_->strides[i] = stride[i];
    }

    memref_wrapper_pointer2_ = reinterpret_cast<MemRefWrapper<N>**>(malloc(sizeof(MemRefWrapper<N>*)));
    *memref_wrapper_pointer2_ = memref_wrapper_pointer_;
  }

  InputTensorWrapper(void* ptr, const std::vector<int64_t>& shape) {
    int64_t rank = shape.size();
    if (rank != N) {
      LOG(FATAL) << "Input ptr's shape vs input shape N, " << rank << " : " << N;
    }
    memref_wrapper_pointer_ = AllocateMemrefDescriptor<N>(rank);
    memref_wrapper_pointer_->allocated_ptr = ptr;
    memref_wrapper_pointer_->aligned_ptr = ptr;
    std::vector<int64_t> stride = MakeStrides(shape);
    for (auto i = 0; i < rank; ++i) {
      memref_wrapper_pointer_->sizes[i] = shape[i];
      memref_wrapper_pointer_->strides[i] = stride[i];
    }

    memref_wrapper_pointer2_ = reinterpret_cast<MemRefWrapper<N>**>(malloc(sizeof(MemRefWrapper<N>*)));
    *memref_wrapper_pointer2_ = memref_wrapper_pointer_;
  }

  void* GetArg() {
    return (void*)memref_wrapper_pointer2_;
  }

  ~InputTensorWrapper() {
    delete memref_wrapper_pointer_;
    delete memref_wrapper_pointer2_;
    memref_wrapper_pointer_ = nullptr;
    memref_wrapper_pointer2_ = nullptr;
  }

  MemRefWrapper<N>* memref_wrapper_pointer_;
  MemRefWrapper<N>** memref_wrapper_pointer2_;
};

} // namespace mlir
} // namespace mlir_runtime


#ifdef _WIN32
#ifndef MLIR_RUNNER_UTILS_EXPORT
#ifdef mlir_runner_utils_EXPORTS
/* We are building this library */
#define MLIR_RUNNER_UTILS_EXPORT __declspec(dllexport)
#else
/* We are using this library */
#define MLIR_RUNNER_UTILS_EXPORT __declspec(dllimport)
#endif
#endif
#else
#define MLIR_RUNNER_UTILS_EXPORT
#endif

// TODO: 
// put external function here ? now in tensorflow/compiler/mlir/compile_util.h
//
extern "C" MLIR_RUNNER_UTILS_EXPORT void
SetExternalMemref(int64_t* dest, mlir::runtime::MemRefWrapper<1>* src,
                  mlir::runtime::ElementType ele_type) {}

#endif 
