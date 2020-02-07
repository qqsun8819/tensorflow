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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TF_LOWER_TO_LLVM_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TF_LOWER_TO_LLVM_H_

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {

class CallExternalFuncOpLowering : public mlir::ConversionPattern {
 public:
  explicit CallExternalFuncOpLowering(mlir::MLIRContext *context)
      : mlir::ConversionPattern(mlir::TF::CallExternalFuncOp::getOperationName(), 1, context) {}

  mlir::PatternMatchResult matchAndRewrite(mlir::Operation *op,
      llvm::ArrayRef<mlir::Value> operands,
      mlir::ConversionPatternRewriter &rewriter) const override;

 private:
  static mlir::FlatSymbolRefAttr InternalGetOrInsertExternalFunc(
      mlir::PatternRewriter &rewriter,
      mlir::ModuleOp module,
      mlir::LLVM::LLVMDialect *llvmDialect);
};

class Call3dExternalFuncOpLowering : public mlir::ConversionPattern {
 public:
  explicit Call3dExternalFuncOpLowering(mlir::MLIRContext *context)
      : mlir::ConversionPattern(mlir::TF::CallExternalFunc2Op::getOperationName(), 1, context) {}

  mlir::PatternMatchResult matchAndRewrite(mlir::Operation *op,
      llvm::ArrayRef<mlir::Value> operands,
      mlir::ConversionPatternRewriter &rewriter) const override;
};

} // namespace mlir

#endif // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TF_LOWER_TO_LLVM_H_