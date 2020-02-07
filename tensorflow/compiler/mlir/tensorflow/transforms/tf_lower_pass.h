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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TF_LOWER_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TF_LOWER_PASS_H_

#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_lower_to_llvm.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_lower_to_affine.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"

namespace mlir {

struct TFLLVMLoweringPass : public mlir::ModulePass<TFLLVMLoweringPass> {
  void runOnModule() final;
};

struct TFAffineLoweringPass : public mlir::ModulePass<TFAffineLoweringPass> {
  void runOnModule() final;
};

struct TFStandardLoweringPass : public mlir::ModulePass<TFStandardLoweringPass> {
  void runOnModule() final;
};

std::unique_ptr<mlir::Pass> CreateTFLowerToAffinePass();
std::unique_ptr<mlir::Pass> CreateTFLowerToLLVMPass();
std::unique_ptr<mlir::Pass> CreateTFLowerToStdPass();

} // end mlir

#endif // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TF_LOWER_PASS_H_