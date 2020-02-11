#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include "tensorflow/compiler/mlir/compile_util.h"

template <int... Dims> struct StaticSizeMult {
  static constexpr int value = 1;
};
 
template <int N, int... Dims> struct StaticSizeMult<N, Dims...> {
  static constexpr int value = N * StaticSizeMult<Dims...>::value;
};
 
static void printSpace(std::ostream &os, int count) {
  for (int i = 0; i < count; ++i) {
    os << ' ';
  }
}
 
template <typename T, int M, int... Dims> struct VectorDataPrinter {
  static void print(std::ostream &os, const Vector<T, M, Dims...> &val);
};
 
template <typename T, int M, int... Dims>
void VectorDataPrinter<T, M, Dims...>::print(std::ostream &os,
                                             const Vector<T, M, Dims...> &val) {
  static_assert(M > 0, "0 dimensioned tensor");
  static_assert(sizeof(val) == M * StaticSizeMult<Dims...>::value * sizeof(T),
                "Incorrect vector size!");
  // First
  os << "(" << val.vector[0];
  if (M > 1)
    os << ", ";
  if (sizeof...(Dims) > 1)
    os << "\n";
  // Kernel
  for (unsigned i = 1; i + 1 < M; ++i) {
    printSpace(os, 2 * sizeof...(Dims));
    os << val.vector[i] << ", ";
    if (sizeof...(Dims) > 1)
      os << "\n";
  }
  // Last
  printSpace(os, sizeof...(Dims));
  os << val.vector[M - 1] << ")";
}
 
template <typename T, int M, int... Dims>
std::ostream &operator<<(std::ostream &os, const Vector<T, M, Dims...> &v) {
  VectorDataPrinter<T, M, Dims...>::print(os, v);
  return os;
}

template <typename T, int N> struct MemRefDataPrinter {
  static void print(std::ostream &os, T *base, int64_t rank, int64_t offset,
                    int64_t *sizes, int64_t *strides);
  static void printFirst(std::ostream &os, T *base, int64_t rank,
                         int64_t offset, int64_t *sizes, int64_t *strides);
  static void printLast(std::ostream &os, T *base, int64_t rank, int64_t offset,
                        int64_t *sizes, int64_t *strides);
};
 
template <typename T> struct MemRefDataPrinter<T, 0> {
  static void print(std::ostream &os, T *base, int64_t rank, int64_t offset,
                    int64_t *sizes = nullptr, int64_t *strides = nullptr);
};
 
template <typename T, int N>
void MemRefDataPrinter<T, N>::printFirst(std::ostream &os, T *base,
                                         int64_t rank, int64_t offset,
                                         int64_t *sizes, int64_t *strides) {
  os << "[";
  MemRefDataPrinter<T, N - 1>::print(os, base, rank, offset, sizes + 1,
                                     strides + 1);
  // If single element, close square bracket and return early.
  if (sizes[0] <= 1) {
    os << "]";
    return;
  }
  os << ", ";
  if (N > 1)
    os << "\n";
}
 
template <typename T, int N>
void MemRefDataPrinter<T, N>::print(std::ostream &os, T *base, int64_t rank,
                                    int64_t offset, int64_t *sizes,
                                    int64_t *strides) {
  printFirst(os, base, rank, offset, sizes, strides);
  for (unsigned i = 1; i + 1 < sizes[0]; ++i) {
    printSpace(os, rank - N + 1);
    MemRefDataPrinter<T, N - 1>::print(os, base, rank, offset + i * strides[0],
                                       sizes + 1, strides + 1);
    os << ", ";
    if (N > 1)
      os << "\n";
  }
  if (sizes[0] <= 1)
    return;
  printLast(os, base, rank, offset, sizes, strides);
}
 
template <typename T, int N>
void MemRefDataPrinter<T, N>::printLast(std::ostream &os, T *base, int64_t rank,
                                        int64_t offset, int64_t *sizes,
                                        int64_t *strides) {
  printSpace(os, rank - N + 1);
  MemRefDataPrinter<T, N - 1>::print(os, base, rank,
                                     offset + (sizes[0] - 1) * (*strides),
                                     sizes + 1, strides + 1);
  os << "]";
}
 
template <typename T>
void MemRefDataPrinter<T, 0>::print(std::ostream &os, T *base, int64_t rank,
                                    int64_t offset, int64_t *sizes,
                                    int64_t *strides) {
  os << base[offset];
}

template <typename T, int N> void printMemRef(StridedMemRefType<T, N> &M) {
  static_assert(N > 0, "Expected N > 0");
  printMemRefMetaData(std::cout, M);
  std::cout << " data = " << std::endl;
  MemRefDataPrinter<T, N>::print(std::cout, M.data, N, M.offset, M.sizes,
                                 M.strides);
  std::cout << std::endl;
}


extern "C"
void _global_print_memref_1d(
    StridedMemRefType<int32_t, 1> *M2) {
  std::cout << "_global_print_memref_1d called\n";
  printMemRef(*M2);
}

extern "C"
void _global_print_memref_2d(
    StridedMemRefType<int32_t, 2> *M2) {
  std::cout << "_global_print_memref_2d called\n";
  printMemRef(*M2);
}

extern "C"
void _global_print_memref_3d(
    StridedMemRefType<int32_t, 3> *M2) {
  std::cout << "_global_print_memref_3d called\n";
  printMemRef(*M2);
}

extern "C"
void _global_print_memref_1d_i64(
    StridedMemRefType<int64_t, 1> *M2) {
  std::cout << "_global_print_memref_1d_64 called\n";
  printMemRef(*M2);
}

extern "C"
void _global_print_memref_1d_i32(
    StridedMemRefType<int32_t, 1> *M2) {
  std::cout << "_global_print_memref_1d_i64i32 called\n";
  printMemRef(*M2);
}

extern "C"
void _global_print_memref_2d_i64(
    StridedMemRefType<int64_t, 2> *M2) {
  std::cout << "_global_print_memref_2d_64 called\n";
  printMemRef(*M2);
}


extern "C"
void _global_print_memref_3d_i64(
    StridedMemRefType<int64_t, 3> *M2) {
  std::cout << "_global_print_memref_3d_64 called\n";
  printMemRef(*M2);
}

extern "C"
void _global_print_memref_2d_i1(
    StridedMemRefType<bool, 2> *M2) {
  std::cout << "_global_print_memref_2d_i1 called\n";
  printMemRef(*M2);
}

extern "C"
void _global_print_memref_2d_f32(
    StridedMemRefType<float, 2> *M2) {
  std::cout << "_global_print_memref_2d_f32 called\n";
  printMemRef(*M2);
}

extern "C"
void _global_print_memref_2d_f64(
    StridedMemRefType<double, 2> *M2) {
  std::cout << "_global_print_memref_2d_f64 called\n";
  printMemRef(*M2);
}


extern "C"
int64_t _global_get_unique_ids_count(
    StridedMemRefType<int64_t, 1> *ids, /*StridedMemRefType<int64_t, 0> *N*/int64_t N) {
  int64_t unique_count = 0;
  int64_t *data = ids->data;
  std::unordered_set<int64_t> m;
  int64_t real_n = N; //*((int64_t*)(N->data));
  for (int64_t i = 0; i < real_n; ++i) {
    if (m.find(*(data+i)) == m.end()) {
      ++unique_count;
      m.insert(*(data+i));
    }
  }

  return unique_count;
}

extern "C"
void _global_unique_ids(
    StridedMemRefType<int64_t, 1> *input_ids,
    StridedMemRefType<int64_t, 0> *id_count,
    StridedMemRefType<int64_t, 1> *output_ids) {
  int32_t cur_idx = -1;
  std::unordered_map<int64_t, int64_t> m;
  int64_t real_n = input_ids->sizes[0]; 
  for (int64_t i = 0; i < real_n; ++i) {
    int64_t spec_id = (*(input_ids->data + i));
    if (m.find(spec_id) == m.end()) {
      m[spec_id] = ++cur_idx;
      *(output_ids->data + cur_idx) = spec_id;
    } else {
      continue;
    }
  }
}

extern "C"
void _global_unique_index32(
    StridedMemRefType<int64_t, 1> *ids,
    StridedMemRefType<int64_t, 1> *unique_ids,
    StridedMemRefType<int32_t, 1> *ids_index) {
  std::unordered_map<int64_t, int32_t> m;
  int64_t unique_N = unique_ids->sizes[0]; 

  for (int32_t i = 0; i < unique_N; ++i) {
    m[*(unique_ids->data + i)] = i;
  }
  int64_t input_N = ids->sizes[0]; 
  for(int64_t i = 0; i < input_N; i++) {
    *(ids_index->data + i) = m[*(ids->data + i)];
  }
}

extern "C"
void _global_unique_index64(
    StridedMemRefType<int64_t, 1> *ids,
    StridedMemRefType<int64_t, 1> *unique_ids,
    StridedMemRefType<int64_t, 1> *ids_index) {
  std::unordered_map<int64_t, int64_t> m;
  int64_t unique_N = unique_ids->sizes[0]; 

  for (int64_t i = 0; i < unique_N; ++i) {
    m[*(unique_ids->data + i)] = i;
  }
  int64_t input_N = ids->sizes[0]; 
  for(int64_t i = 0; i < input_N; i++) {
    *(ids_index->data + i) = m[*(ids->data + i)];
  }
}

extern "C" void
_global_set_external_memref_r0_i32(int64_t dest, RRawMemRef<0>* src) {
  set_external_memref<0>(dest, src, RElementType::I32Type);
}
extern "C" void
_global_set_external_memref_r1_i32(int64_t dest, RRawMemRef<1>* src) {
  set_external_memref<1>(dest, src, RElementType::I32Type);
}
extern "C" void
_global_set_external_memref_r2_i32(int64_t dest, RRawMemRef<2>* src) {
  set_external_memref<2>(dest, src, RElementType::I32Type);
}
extern "C" void
_global_set_external_memref_r3_i32(int64_t dest, RRawMemRef<3>* src) {
  set_external_memref<3>(dest, src, RElementType::I32Type);
}
extern "C" void
_global_set_external_memref_r4_i32(int64_t dest, RRawMemRef<4>* src) {
  set_external_memref<4>(dest, src, RElementType::I32Type);
}
extern "C" void
_global_set_external_memref_r5_i32(int64_t dest, RRawMemRef<5>* src) {
  set_external_memref<5>(dest, src, RElementType::I32Type);
}

extern "C" void
_global_set_external_memref_r0_i64(int64_t dest, RRawMemRef<0>* src) {
  set_external_memref<0>(dest, src, RElementType::I64Type);
}
extern "C" void
_global_set_external_memref_r1_i64(int64_t dest, RRawMemRef<1>* src) {
  set_external_memref<1>(dest, src, RElementType::I64Type);
}
extern "C" void
_global_set_external_memref_r2_i64(int64_t dest, RRawMemRef<2>* src) {
  set_external_memref<2>(dest, src, RElementType::I64Type);
}
extern "C" void
_global_set_external_memref_r3_i64(int64_t dest, RRawMemRef<3>* src) {
  set_external_memref<3>(dest, src, RElementType::I64Type);
}
extern "C" void
_global_set_external_memref_r4_i64(int64_t dest, RRawMemRef<4>* src) {
  set_external_memref<4>(dest, src, RElementType::I64Type);
}
extern "C" void
_global_set_external_memref_r5_i64(int64_t dest, RRawMemRef<5>* src) {
  set_external_memref<5>(dest, src, RElementType::I64Type);
}

extern "C" void
_global_set_external_memref_r0_f32(int64_t dest, RRawMemRef<0>* src) {
  set_external_memref<0>(dest, src, RElementType::F32Type);
}
extern "C" void
_global_set_external_memref_r1_f32(int64_t dest, RRawMemRef<1>* src) {
  set_external_memref<1>(dest, src, RElementType::F32Type);
}
extern "C" void
_global_set_external_memref_r2_f32(int64_t dest, RRawMemRef<2>* src) {
  set_external_memref<2>(dest, src, RElementType::F32Type);
}
extern "C" void
_global_set_external_memref_r3_f32(int64_t dest, RRawMemRef<3>* src) {
  set_external_memref<3>(dest, src, RElementType::F32Type);
}
extern "C" void
_global_set_external_memref_r4_f32(int64_t dest, RRawMemRef<4>* src) {
  set_external_memref<4>(dest, src, RElementType::F32Type);
}
extern "C" void
_global_set_external_memref_r5_f32(int64_t dest, RRawMemRef<5>* src) {
  set_external_memref<5>(dest, src, RElementType::F32Type);
}

extern "C" void
_global_set_external_memref_r0_f64(int64_t dest, RRawMemRef<0>* src) {
  set_external_memref<0>(dest, src, RElementType::F64Type);
}
extern "C" void
_global_set_external_memref_r1_f64(int64_t dest, RRawMemRef<1>* src) {
  set_external_memref<1>(dest, src, RElementType::F64Type);
}
extern "C" void
_global_set_external_memref_r2_f64(int64_t dest, RRawMemRef<2>* src) {
  set_external_memref<2>(dest, src, RElementType::F64Type);
}
extern "C" void
_global_set_external_memref_r3_f64(int64_t dest, RRawMemRef<3>* src) {
  set_external_memref<3>(dest, src, RElementType::F64Type);
}
extern "C" void
_global_set_external_memref_r4_f64(int64_t dest, RRawMemRef<4>* src) {
  set_external_memref<4>(dest, src, RElementType::F64Type);
}
extern "C" void
_global_set_external_memref_r5_f64(int64_t dest, RRawMemRef<5>* src) {
  set_external_memref<5>(dest, src, RElementType::F64Type);
}

