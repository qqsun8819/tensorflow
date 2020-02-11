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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_LOWER_TO_AFFINE_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_LOWER_TO_AFFINE_H_

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {

struct ConstOpLowering : public mlir::OpRewritePattern<mlir::TF::ConstOp> {
  using mlir::OpRewritePattern<mlir::TF::ConstOp>::OpRewritePattern;
  
  mlir::PatternMatchResult matchAndRewrite(
      mlir::TF::ConstOp op,
      mlir::PatternRewriter &rewriter) const final;
};

class UniqueOpLOwering : public OpRewritePattern<mlir::TF::UniqueOp> {
 public:
  using mlir::OpRewritePattern<mlir::TF::UniqueOp>::OpRewritePattern;

  mlir::PatternMatchResult matchAndRewrite(
      mlir::TF::UniqueOp op,
      mlir::PatternRewriter &rewriter) const final;
};

class ReshapeOpLOwering : public OpRewritePattern<mlir::TF::ReshapeOp> {
 public:
  using mlir::OpRewritePattern<mlir::TF::ReshapeOp>::OpRewritePattern;

  mlir::PatternMatchResult matchAndRewrite(
      mlir::TF::ReshapeOp op,
      mlir::PatternRewriter &rewriter) const final;
};

class CopyResultOpLowering : public OpRewritePattern<mlir::TF::CopyResultOp> {
 public:
  using mlir::OpRewritePattern<mlir::TF::CopyResultOp>::OpRewritePattern;

  mlir::PatternMatchResult matchAndRewrite(
      mlir::TF::CopyResultOp op,
      mlir::PatternRewriter &rewriter) const final;
};

} // end mlir

#endif // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_LOWER_TO_AFFINE_H_
