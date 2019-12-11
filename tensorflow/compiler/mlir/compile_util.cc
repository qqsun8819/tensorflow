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
void _global_mlir_call_external_func_1d(
    StridedMemRefType<int32_t, 1> *M1,
    StridedMemRefType<int32_t, 1> *M2) {
  printMemRef(*M1);
  printMemRef(*M2);
}

extern "C"
void _global_mlir_call_external_func_3d(
    StridedMemRefType<int32_t, 3> *M1,
    StridedMemRefType<int32_t, 3> *M2) {
  printMemRef(*M1);
  printMemRef(*M2);
}

