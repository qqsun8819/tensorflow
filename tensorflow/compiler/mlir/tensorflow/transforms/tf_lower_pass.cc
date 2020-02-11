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

#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_lower_pass.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/TargetSelect.h"

namespace mlir {

void TFAffineLoweringPass::runOnModule() {
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::AffineOpsDialect,
                         mlir::StandardOpsDialect>();
  target.addIllegalDialect<mlir::TF::TensorFlowDialect>();
  mlir::LLVMTypeConverter type_converter(&getContext());
  target.addLegalOp<mlir::TF::MemcpyOp>();
  // NOTE(jiankeng.pt): Advanced skills: prevent the error of UniqueOp lowering process.
  // Cause the `func` op here has no one legalize pattern.
  // If you don't want the Op with concrete information which you specify
  // in the anonymous function be lowered at the pass, please do this.
  target.addDynamicallyLegalOp<FuncOp>(
      [&](FuncOp op) { return  op.getName() == "_global_unique" ||
                               op.getName() == "_global_unique64" ||
                               op.getName() == "_global_get_unique_ids_count" ||
                               op.getName() == "_global_mlir_call_external_func_64" ||
                               op.getName() == "_global_mlir_call_external_func_1d_i64" ||
                               op.getName() == "_global_mlir_call_external_func_1d_i64i32" ||
                               op.getName() == "_global_mlir_call_external_func_1d" ||
                               op.getName() == "_global_mlir_call_external_func_2d" ||
                               op.getName() == "_global_mlir_call_external_func_2d_i64" ||
                               op.getName() == "_global_mlir_call_external_func_2d_f64" ||
                               op.getName() == "_global_mlir_call_external_func_2d_f32" ||
                               op.getName() == "_global_mlir_call_external_func_2d_i1" ||
                               op.getName() == "_global_set_external_memref_r0_i32" ||
                               op.getName() == "_global_set_external_memref_r1_i32" ||
                               op.getName() == "_global_set_external_memref_r2_i32" ||
                               op.getName() == "_global_set_external_memref_r3_i32" ||
                               op.getName() == "_global_set_external_memref_r4_i32" ||
                               op.getName() == "_global_set_external_memref_r5_i32" ||
                               op.getName() == "_global_set_external_memref_r0_i64" ||
                               op.getName() == "_global_set_external_memref_r1_i64" ||
                               op.getName() == "_global_set_external_memref_r2_i64" ||
                               op.getName() == "_global_set_external_memref_r3_i64" ||
                               op.getName() == "_global_set_external_memref_r4_i64" ||
                               op.getName() == "_global_set_external_memref_r5_i64" ||
                               op.getName() == "_global_set_external_memref_r0_f32" ||
                               op.getName() == "_global_set_external_memref_r1_f32" ||
                               op.getName() == "_global_set_external_memref_r2_f32" ||
                               op.getName() == "_global_set_external_memref_r3_f32" ||
                               op.getName() == "_global_set_external_memref_r4_f32" ||
                               op.getName() == "_global_set_external_memref_r5_f32" ||
                               op.getName() == "_global_set_external_memref_r0_f64" ||
                               op.getName() == "_global_set_external_memref_r1_f64" ||
                               op.getName() == "_global_set_external_memref_r2_f64" ||
                               op.getName() == "_global_set_external_memref_r3_f64" ||
                               op.getName() == "_global_set_external_memref_r4_f64" ||
                               op.getName() == "_global_set_external_memref_r5_f64"; });

  mlir::OwningRewritePatternList patterns;
  patterns.insert<ConstOpLowering>(&getContext());
  patterns.insert<CopyResultOpLowering>(&getContext());

  auto module = getModule();
  for (auto func : module.getOps<mlir::FuncOp>()) {
    if (failed(mlir::applyPartialConversion(func, target, patterns))) {
      signalPassFailure();
    }
  }
  /*
  auto module = getModule();
  if (mlir::failed(mlir::applyFullConversion(module, target,
                                             patterns, &type_converter))) {
    signalPassFailure();
  }*/
}

void TFStandardLoweringPass::runOnModule() {
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::LLVM::LLVMDialect,
                         mlir::StandardOpsDialect>();
  target.addLegalOp<mlir::ModuleOp, mlir::FuncOp,
                    mlir::ModuleTerminatorOp>();
  target.addIllegalDialect<mlir::TF::TensorFlowDialect>();
  target.addLegalOp<mlir::TF::CallExternalFuncOp>();
  target.addLegalOp<mlir::TF::CallExternalFunc2Op>();

  mlir::LLVMTypeConverter type_converter(&getContext());
  mlir::OwningRewritePatternList patterns;
  mlir::populateAffineToStdConversionPatterns(patterns, &getContext());

  auto module = getModule();
  for (auto func : module.getOps<mlir::FuncOp>()) {
    if (mlir::failed(mlir::applyFullConversion(func, target,
                                               patterns, &type_converter))) {
      signalPassFailure();
    }
  }
}

void TFLLVMLoweringPass::runOnModule() {
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
  target.addIllegalDialect<mlir::TF::TensorFlowDialect>();

  mlir::LLVMTypeConverter type_converter(&getContext());

  mlir::OwningRewritePatternList patterns;
  mlir::populateAffineToStdConversionPatterns(patterns, &getContext());
  mlir::populateLoopToStdConversionPatterns(patterns, &getContext());
  mlir::populateStdToLLVMConversionPatterns(type_converter, patterns);

  patterns.insert<CallExternalFuncOpLowering>(&getContext());
  patterns.insert<Call3dExternalFuncOpLowering>(&getContext());

  // TODO: here use module pass!
  auto module = getModule();
  if (mlir::failed(mlir::applyFullConversion(module, target,
                                             patterns, &type_converter))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> CreateTFLowerToAffinePass() {
  return std::make_unique<TFAffineLoweringPass>();
}

std::unique_ptr<mlir::Pass> CreateTFLowerToStdPass() {
  return std::make_unique<TFStandardLoweringPass>();
}

std::unique_ptr<mlir::Pass> CreateTFLowerToLLVMPass() {
  return std::make_unique<TFLLVMLoweringPass>();
}


} // namespace mlir

static mlir::PassRegistration<mlir::TFAffineLoweringPass> pass1(
        "convert-tf-to-affine",
            "Convert from TF dialect to affine dialect");

static mlir::PassRegistration<mlir::TFLLVMLoweringPass> pass2(
        "convert-tf-to-llvm",
            "Convert from TF dialect to affine dialect");

static mlir::PassRegistration<mlir::TFStandardLoweringPass> pass3(
        "convert-tf-to-std",
            "Convert from TF dialect to std dialect");
