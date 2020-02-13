#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include "tensorflow/compiler/mlir/external_copy_results.h"


extern "C" int64_t
_global_set_external_memref_r0_i32(RRawMemRef<1>* dest, RRawMemRef<0>* src) {
  return set_external_memref<0>(dest, src, RElementType::I32Type);
}
extern "C" int64_t
_global_set_external_memref_r1_i32(RRawMemRef<1>* dest, RRawMemRef<1>* src) {
  return set_external_memref<1>(dest, src, RElementType::I32Type);
}
extern "C" int64_t
_global_set_external_memref_r2_i32(RRawMemRef<1>* dest, RRawMemRef<2>* src) {
  return set_external_memref<2>(dest, src, RElementType::I32Type);
}
extern "C" int64_t
_global_set_external_memref_r3_i32(RRawMemRef<1>* dest, RRawMemRef<3>* src) {
  return set_external_memref<3>(dest, src, RElementType::I32Type);
}
extern "C" int64_t
_global_set_external_memref_r4_i32(RRawMemRef<1>* dest, RRawMemRef<4>* src) {
  return set_external_memref<4>(dest, src, RElementType::I32Type);
}
extern "C" int64_t
_global_set_external_memref_r5_i32(RRawMemRef<1>* dest, RRawMemRef<5>* src) {
  return set_external_memref<5>(dest, src, RElementType::I32Type);
}

extern "C" int64_t
_global_set_external_memref_r0_i64(RRawMemRef<1>* dest, RRawMemRef<0>* src) {
  return set_external_memref<0>(dest, src, RElementType::I64Type);
}
extern "C" int64_t
_global_set_external_memref_r1_i64(RRawMemRef<1>* dest, RRawMemRef<1>* src) {
  return set_external_memref<1>(dest, src, RElementType::I64Type);
}
extern "C" int64_t
_global_set_external_memref_r2_i64(RRawMemRef<1>* dest, RRawMemRef<2>* src) {
  return set_external_memref<2>(dest, src, RElementType::I64Type);
}
extern "C" int64_t
_global_set_external_memref_r3_i64(RRawMemRef<1>* dest, RRawMemRef<3>* src) {
  return set_external_memref<3>(dest, src, RElementType::I64Type);
}
extern "C" int64_t
_global_set_external_memref_r4_i64(RRawMemRef<1>* dest, RRawMemRef<4>* src) {
  return set_external_memref<4>(dest, src, RElementType::I64Type);
}
extern "C" int64_t
_global_set_external_memref_r5_i64(RRawMemRef<1>* dest, RRawMemRef<5>* src) {
  return set_external_memref<5>(dest, src, RElementType::I64Type);
}

extern "C" int64_t
_global_set_external_memref_r0_f32(RRawMemRef<1>* dest, RRawMemRef<0>* src) {
  return set_external_memref<0>(dest, src, RElementType::F32Type);
}
extern "C" int64_t
_global_set_external_memref_r1_f32(RRawMemRef<1>* dest, RRawMemRef<1>* src) {
  return set_external_memref<1>(dest, src, RElementType::F32Type);
}
extern "C" int64_t
_global_set_external_memref_r2_f32(RRawMemRef<1>* dest, RRawMemRef<2>* src) {
  return set_external_memref<2>(dest, src, RElementType::F32Type);
}
extern "C" int64_t
_global_set_external_memref_r3_f32(RRawMemRef<1>* dest, RRawMemRef<3>* src) {
  return set_external_memref<3>(dest, src, RElementType::F32Type);
}
extern "C" int64_t
_global_set_external_memref_r4_f32(RRawMemRef<1>* dest, RRawMemRef<4>* src) {
  return set_external_memref<4>(dest, src, RElementType::F32Type);
}
extern "C" int64_t
_global_set_external_memref_r5_f32(RRawMemRef<1>* dest, RRawMemRef<5>* src) {
  return set_external_memref<5>(dest, src, RElementType::F32Type);
}

extern "C" int64_t
_global_set_external_memref_r0_f64(RRawMemRef<1>* dest, RRawMemRef<0>* src) {
  return set_external_memref<0>(dest, src, RElementType::F64Type);
}
extern "C" int64_t
_global_set_external_memref_r1_f64(RRawMemRef<1>* dest, RRawMemRef<1>* src) {
  return set_external_memref<1>(dest, src, RElementType::F64Type);
}
extern "C" int64_t
_global_set_external_memref_r2_f64(RRawMemRef<1>* dest, RRawMemRef<2>* src) {
  return set_external_memref<2>(dest, src, RElementType::F64Type);
}
extern "C" int64_t
_global_set_external_memref_r3_f64(RRawMemRef<1>* dest, RRawMemRef<3>* src) {
  return set_external_memref<3>(dest, src, RElementType::F64Type);
}
extern "C" int64_t
_global_set_external_memref_r4_f64(RRawMemRef<1>* dest, RRawMemRef<4>* src) {
  return set_external_memref<4>(dest, src, RElementType::F64Type);
}
extern "C" int64_t
_global_set_external_memref_r5_f64(RRawMemRef<1>* dest, RRawMemRef<5>* src) {
  return set_external_memref<5>(dest, src, RElementType::F64Type);
}

