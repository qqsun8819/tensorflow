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

#ifndef TENSORFLOW_COMPILER_JIT_KERNELS_MLIR_OPS_H_
#define TENSORFLOW_COMPILER_JIT_KERNELS_MLIR_OPS_H_

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/stream_executor_util.h"
#include "tensorflow/stream_executor/tf_allocator_adapter.h"

namespace tensorflow {

class MlirRunOp : public OpKernel {
 public:
  explicit MlirRunOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

 private:
  std::string entry_func_name_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_KERNELS_MLIR_OPS_H_

