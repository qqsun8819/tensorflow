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

#ifndef TENSORFLOW_COMPILER_MLIR_COMPILE_UTIL_H
#define TENSORFLOW_COMPILER_MLIR_COMPILE_UTIL_H

#include <assert.h>
#include <cstdint>
#include <string>
#include <iostream>

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

template <typename T, int N> struct StridedMemRefType;
template <typename StreamType, typename T, int N>
void printMemRefMetaData(StreamType &os, StridedMemRefType<T, N> &V);

template <int N> void dropFront(int64_t arr[N], int64_t *res) {
  for (unsigned i = 1; i < N; ++i)
    *(res + i - 1) = arr[i];
}

/// StridedMemRef descriptor type with static rank.
template <typename T, int N> struct StridedMemRefType {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
  // This operator[] is extremely slow and only for sugaring purposes.
  StridedMemRefType<T, N - 1> operator[](int64_t idx) {
    StridedMemRefType<T, N - 1> res;
    res.basePtr = basePtr;
    res.data = data;
    res.offset = offset + idx * strides[0];
    dropFront<N>(sizes, res.sizes);
    dropFront<N>(strides, res.strides);
    return res;
  }
};

/// StridedMemRef descriptor type specialized for rank 1.
template <typename T> struct StridedMemRefType<T, 1> {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
  T &operator[](int64_t idx) { return *(data + offset + idx * strides[0]); } };

/// StridedMemRef descriptor type specialized for rank 0.
template <typename T> struct StridedMemRefType<T, 0> {
  T *basePtr;
  T *data;
  int64_t offset;
};

template <typename StreamType, typename T, int N>
void printMemRefMetaData(StreamType &os, StridedMemRefType<T, N> &V) {
  static_assert(N > 0, "Expected N > 0");
  os << "Memref base@ = " << V.data << " rank = " << N
     << " offset = " << V.offset << " sizes = [" << V.sizes[0];
  for (unsigned i = 1; i < N; ++i)
    os << ", " << V.sizes[i];
  os << "] strides = [" << V.strides[0];
  for (unsigned i = 1; i < N; ++i)
    os << ", " << V.strides[i];
  os << "]";
}

template <typename StreamType, typename T>
void printMemRefMetaData(StreamType &os, StridedMemRefType<T, 0> &V) {
  os << "Memref base@ = " << V.data << " rank = 0"
     << " offset = " << V.offset;
}

template <typename T, int Dim, int... Dims> struct Vector {
  Vector<T, Dims...> vector[Dim];
};
template <typename T, int Dim> struct Vector<T, Dim> { T vector[Dim]; };

template <int D1, typename T> using Vector1D = Vector<T, D1>;
template <int D1, int D2, typename T> using Vector2D = Vector<T, D1, D2>;
template <int D1, int D2, int D3, typename T>
using Vector3D = Vector<T, D1, D2, D3>;
template <int D1, int D2, int D3, int D4, typename T>
using Vector4D = Vector<T, D1, D2, D3, D4>;

extern "C" MLIR_RUNNER_UTILS_EXPORT void
_global_print_memref_1d(
    StridedMemRefType<int32_t, 1> *M2);

extern "C" MLIR_RUNNER_UTILS_EXPORT void
_global_print_memref_2d(
    StridedMemRefType<int32_t, 2> *M2);

extern "C" MLIR_RUNNER_UTILS_EXPORT void
_global_print_memref_3d(
    StridedMemRefType<int32_t, 3> *M2);

extern "C" MLIR_RUNNER_UTILS_EXPORT void
_global_print_memref_1d_i64(
    StridedMemRefType<int64_t, 1> *M2);

// TODO: Refine these code!
//
extern "C" MLIR_RUNNER_UTILS_EXPORT void
_global_print_memref_1d_i32(
    StridedMemRefType<int32_t, 1> *M2);

extern "C" MLIR_RUNNER_UTILS_EXPORT void
_global_print_memref_2d_i64(
    StridedMemRefType<int64_t, 2> *M2);

extern "C" MLIR_RUNNER_UTILS_EXPORT void
_global_print_memref_3d_i64(
    StridedMemRefType<int64_t, 3> *M2);

extern "C" MLIR_RUNNER_UTILS_EXPORT void
_global_print_memref_2d_f32(
    StridedMemRefType<float, 2> *M2);

extern "C" MLIR_RUNNER_UTILS_EXPORT void
_global_print_memref_2d_f64(
    StridedMemRefType<double, 2> *M2);

extern "C" MLIR_RUNNER_UTILS_EXPORT void
_global_print_memref_2d_i1(
    StridedMemRefType<bool, 2> *M2);


// TODO: UniqueOp support int32 and int64 for ids index
//

extern "C" MLIR_RUNNER_UTILS_EXPORT int64_t
_global_get_unique_ids_count(StridedMemRefType<int64_t, 1> *ids, int64_t N);

extern "C" MLIR_RUNNER_UTILS_EXPORT 
void _global_unique_ids(
    StridedMemRefType<int64_t, 1> *intput_ids,
    StridedMemRefType<int64_t, 0> *id_count,
    StridedMemRefType<int64_t, 1> *output_ids); 

extern "C" MLIR_RUNNER_UTILS_EXPORT 
void _global_unique_index32(
    StridedMemRefType<int64_t, 1> *ids,
    StridedMemRefType<int64_t, 1> *unique_ids,
    StridedMemRefType<int32_t, 1> *ids_index); 

extern "C" MLIR_RUNNER_UTILS_EXPORT 
void _global_unique_index64(
    StridedMemRefType<int64_t, 1> *ids,
    StridedMemRefType<int64_t, 1> *unique_ids,
    StridedMemRefType<int64_t, 1> *ids_index); 


#endif TENSORFLOW_COMPILER_MLIR_COMPILE_UTIL_H
