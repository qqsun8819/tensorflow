/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_EXTERNAL_COPY_RESULTS_H
#define TENSORFLOW_COMPILER_MLIR_EXTERNAL_COPY_RESULTS_H

#include <assert.h>
#include <cstdint>
#include <string>
#include <iostream>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

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

// ----------------------------------------------------------
// For get result from compiled function
//

template <int N>
struct RRawMemRef {
  void* allocated_ptr;
  void* aligned_ptr;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

typedef enum {
  I32Type = 0,
  I64Type = 1,
  F32Type = 2,
  F64Type = 3,

  InvalidType = 100,
} RElementType;

template <int N>
int64_t set_external_memref(RRawMemRef<1>* dest, RRawMemRef<N>* src, int ele_type) {
  int64_t dest_addr = *((int64_t*)(dest->aligned_ptr));
  int64_t* dest_addr_addr = (int64_t*)(dest_addr);

  // TODO: consider strides

  int64_t total_count = 1;
  std::vector<tensorflow::int64> dim_sizes;
  for (int i = 0; i < N; ++i) {
    dim_sizes.push_back(*(src->sizes+i));
    total_count *= *(src->sizes+i);
  }

  tensorflow::DataType dt;
  if (ele_type == RElementType::I32Type) {
    dt = tensorflow::DT_INT32;
  } else if (ele_type == RElementType::I64Type) {
    dt = tensorflow::DT_INT64;
  } else if (ele_type == RElementType::F32Type) {
    dt = tensorflow::DT_FLOAT;
  } else if (ele_type == RElementType::F64Type) {
    dt = tensorflow::DT_DOUBLE;
  } else {
    assert(false && "Now only support i32, i64, f32, f64 type tensor.");
  }
  tensorflow::Tensor* tmp_tensor(new tensorflow::Tensor(dt, tensorflow::TensorShape(dim_sizes)));
  *dest_addr_addr = (int64_t)(tmp_tensor);

  // TODO: should be optimized ?
  for (int64_t i = 0; i < total_count; ++i) {
    switch (ele_type) {
      case RElementType::I32Type: tmp_tensor->flat<tensorflow::int32>()(i) = *((int32_t*)(src->aligned_ptr)+i); break;
      case RElementType::I64Type: tmp_tensor->flat<tensorflow::int64>()(i) = *((int64_t*)(src->aligned_ptr)+i); break;
      case RElementType::F32Type: tmp_tensor->flat<float>()(i) = *((float*)(src->aligned_ptr)+i); break;
      case RElementType::F64Type: tmp_tensor->flat<double>()(i) = *((double*)(src->aligned_ptr)+i); break;
    }
  }

  return 0;
}

// Rank=0
extern "C" MLIR_RUNNER_UTILS_EXPORT int64_t
_global_set_external_memref_r0_i32(RRawMemRef<1>* dest, RRawMemRef<0>* src);
// Rank=1
extern "C" MLIR_RUNNER_UTILS_EXPORT int64_t
_global_set_external_memref_r1_i32(RRawMemRef<1>* dest, RRawMemRef<1>* src);
// Rank=2
extern "C" MLIR_RUNNER_UTILS_EXPORT int64_t
_global_set_external_memref_r2_i32(RRawMemRef<1>* dest, RRawMemRef<2>* src);
// Rank=3
extern "C" MLIR_RUNNER_UTILS_EXPORT int64_t
_global_set_external_memref_r3_i32(RRawMemRef<1>* dest, RRawMemRef<3>* src);
// Rank=4
extern "C" MLIR_RUNNER_UTILS_EXPORT int64_t
_global_set_external_memref_r4_i32(RRawMemRef<1>* dest, RRawMemRef<4>* src);
// Rank=5
extern "C" MLIR_RUNNER_UTILS_EXPORT int64_t
_global_set_external_memref_r5_i32(RRawMemRef<1>* dest, RRawMemRef<5>* src);

// Rank=0
extern "C" MLIR_RUNNER_UTILS_EXPORT int64_t
_global_set_external_memref_r0_i64(RRawMemRef<1>* dest, RRawMemRef<0>* src);
// Rank=1
extern "C" MLIR_RUNNER_UTILS_EXPORT int64_t
_global_set_external_memref_r1_i64(RRawMemRef<1>* dest, RRawMemRef<1>* src);
// Rank=2
extern "C" MLIR_RUNNER_UTILS_EXPORT int64_t
_global_set_external_memref_r2_i64(RRawMemRef<1>* dest, RRawMemRef<2>* src);
// Rank=3
extern "C" MLIR_RUNNER_UTILS_EXPORT int64_t
_global_set_external_memref_r3_i64(RRawMemRef<1>* dest, RRawMemRef<3>* src);
// Rank=4
extern "C" MLIR_RUNNER_UTILS_EXPORT int64_t
_global_set_external_memref_r4_i64(RRawMemRef<1>* dest, RRawMemRef<4>* src);
// Rank=5
extern "C" MLIR_RUNNER_UTILS_EXPORT int64_t
_global_set_external_memref_r5_i64(RRawMemRef<1>* dest, RRawMemRef<5>* src);

// Rank=0
extern "C" MLIR_RUNNER_UTILS_EXPORT int64_t
_global_set_external_memref_r0_f32(RRawMemRef<1>* dest, RRawMemRef<0>* src);
// Rank=1
extern "C" MLIR_RUNNER_UTILS_EXPORT int64_t
_global_set_external_memref_r1_f32(RRawMemRef<1>* dest, RRawMemRef<1>* src);
// Rank=2
extern "C" MLIR_RUNNER_UTILS_EXPORT int64_t
_global_set_external_memref_r2_f32(RRawMemRef<1>* dest, RRawMemRef<2>* src);
// Rank=3
extern "C" MLIR_RUNNER_UTILS_EXPORT int64_t
_global_set_external_memref_r3_f32(RRawMemRef<1>* dest, RRawMemRef<3>* src);
// Rank=4
extern "C" MLIR_RUNNER_UTILS_EXPORT int64_t
_global_set_external_memref_r4_f32(RRawMemRef<1>* dest, RRawMemRef<4>* src);
// Rank=5
extern "C" MLIR_RUNNER_UTILS_EXPORT int64_t
_global_set_external_memref_r5_f32(RRawMemRef<1>* dest, RRawMemRef<5>* src);

// Rank=0
extern "C" MLIR_RUNNER_UTILS_EXPORT int64_t
_global_set_external_memref_r0_f64(RRawMemRef<1>* dest, RRawMemRef<0>* src);
// Rank=1
extern "C" MLIR_RUNNER_UTILS_EXPORT int64_t
_global_set_external_memref_r1_f64(RRawMemRef<1>* dest, RRawMemRef<1>* src);
// Rank=2
extern "C" MLIR_RUNNER_UTILS_EXPORT int64_t
_global_set_external_memref_r2_f64(RRawMemRef<1>* dest, RRawMemRef<2>* src);
// Rank=3
extern "C" MLIR_RUNNER_UTILS_EXPORT int64_t
_global_set_external_memref_r3_f64(RRawMemRef<1>* dest, RRawMemRef<3>* src);
// Rank=4
extern "C" MLIR_RUNNER_UTILS_EXPORT int64_t
_global_set_external_memref_r4_f64(RRawMemRef<1>* dest, RRawMemRef<4>* src);
// Rank=5
extern "C" MLIR_RUNNER_UTILS_EXPORT int64_t
_global_set_external_memref_r5_f64(RRawMemRef<1>* dest, RRawMemRef<5>* src);


#endif TENSORFLOW_COMPILER_MLIR_EXTERNAL_COPY_RESULTS_H
