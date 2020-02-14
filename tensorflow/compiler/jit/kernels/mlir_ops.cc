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

#include <vector>

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "tensorflow/compiler/jit/kernels/mlir_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/runtime/dynamic_memref.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

// Now support the max tensor rank is 5
const int kMaxRank = 5;

namespace {

void* BuildCompiledFunctionArgs(
  std::vector<std::vector<void*>>& args_resource_saver,
  Tensor t) {
  int rank = t.dims();
  if (rank > kMaxRank) {
    LOG(FATAL) << "Not support rank, the supported max rank is 5, current is "
               << rank;
  }

  switch (rank) {
#define NEW_INPUT_TENSOR_ARG(R)                        \
  case R:                                              \
    {                                                  \
      mlir::runtime::InputTensorWrapper<R>* arg =      \
          new mlir::runtime::InputTensorWrapper<R>(t); \
      args_resource_saver[R].push_back((void*)(arg));  \
      return arg->GetArg();                            \
    }

    NEW_INPUT_TENSOR_ARG(0);
    NEW_INPUT_TENSOR_ARG(1);
    NEW_INPUT_TENSOR_ARG(2);
    NEW_INPUT_TENSOR_ARG(3);
    NEW_INPUT_TENSOR_ARG(4);
    NEW_INPUT_TENSOR_ARG(5);
#undef NEW_INPUT_TENSOR_ARG
    default: break;
  }

  return nullptr;
}

void FreeCompiledFunctionArgs(
    std::vector<std::vector<void*>>& args) {
  for (int i = 0; i < args.size(); ++i) {
    switch (i) {
#define FREE_ARGS(R, arg)                                    \
  case R:                                                    \
    for (auto ptr : arg)                                     \
      delete (mlir::runtime::InputTensorWrapper<R>*)(ptr);   \
   break;

      FREE_ARGS(0, args[0]);
      FREE_ARGS(1, args[1]);
      FREE_ARGS(2, args[2]);
      FREE_ARGS(3, args[3]);
      FREE_ARGS(4, args[4]);
      FREE_ARGS(5, args[5]);
#undef FREE_ARGS
      default: break;
    }
  }
}

}

MlirRunOp::MlirRunOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {}

void MlirRunOp::Compute(OpKernelContext* ctx) {
  VLOG(3) << "MlirRunOp " << def().name();

  std::vector<std::vector<void*>> input_args_tmp_src;
  input_args_tmp_src.resize(kMaxRank+1);
  std::vector<mlir::runtime::ResultTensorWrapper*> output_args_tmp_src;

  int input_tensor_num = ctx->num_inputs();
  int output_tensor_num = ctx->num_outputs();
  std::vector<void*> args_pointers;

  // construct input args
  for (int i = 0; i < input_tensor_num; ++i) {
    args_pointers.push_back(
        BuildCompiledFunctionArgs(input_args_tmp_src, ctx->input(i)));
  }

  // construct output result args
  for (int i = 0; i < output_tensor_num; ++i) {
    mlir::runtime::ResultTensorWrapper* rtw =
        new mlir::runtime::ResultTensorWrapper();
    args_pointers.push_back(rtw->GetArg());
    output_args_tmp_src.push_back(rtw);
  }
 
  // TODO: get func name from attr
  std::string func_name = "";
  // TODO: Get cached executable
  mlir::ExecutionEngine* engine = nullptr;
  if (!engine) {
    LOG(FATAL) << "Can not find cached executable, key is " << func_name;
  }
  auto expected_ptr = engine->lookup(func_name);
  if (!expected_ptr) {
    LOG(FATAL) << "Can not find compiled function, func name is " << func_name;
  }

  void (*fptr)(void **) = *expected_ptr;
  (*fptr)((void**)(args_pointers.data()));

  // set tensor to output && free tmp resource
  for (int i = 0; i < output_args_tmp_src.size(); ++i) {
    tensorflow::Tensor* result_tensor =
        output_args_tmp_src[i]->GetResultTensorPointer();
    ctx->set_output(i, *result_tensor);
    delete output_args_tmp_src[i];
  }
  FreeCompiledFunctionArgs(input_args_tmp_src);
}

REGISTER_KERNEL_BUILDER(Name("_MlirRun").Device(DEVICE_CPU), MlirRunOp);

} // namespace tensorflow
