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

#include <iostream>

#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_lower_to_llvm.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "llvm/Support/Debug.h"

namespace mlir {

namespace {

// Insert the external function symbol to module
// pass memref type params to the external function
//
mlir::FlatSymbolRefAttr GetOrInsertMemrefExternalFunc(
    mlir::PatternRewriter &rewriter,
    mlir::ModuleOp module,
    mlir::LLVM::LLVMDialect *llvm_dialect,
    const std::vector<int64_t>& params_dims,
    const std::string& func_name) {
  auto *context = module.getContext();
  if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(func_name)) {
    return mlir::SymbolRefAttr::get(func_name, context);
  }
 
  auto llvm_I64_type = mlir::LLVM::LLVMType::getInt64Ty(llvm_dialect);
  auto llvm_I32_ptr_type = mlir::LLVM::LLVMType::getInt32Ty(llvm_dialect).getPointerTo();
  auto llvm_I32_type = mlir::LLVM::LLVMType::getInt32Ty(llvm_dialect);
  auto llvm_array_1xi64_type = mlir::LLVM::LLVMType::getArrayTy(llvm_I64_type, 1);
  auto int32x1StructPtrType = mlir::LLVM::LLVMType::getStructTy(llvm_dialect,
      {llvm_I32_ptr_type, llvm_I32_ptr_type, llvm_I64_type, llvm_array_1xi64_type, llvm_array_1xi64_type}).getPointerTo();
  SmallVector<LLVM::LLVMType, 4> args_types;
  args_types.reserve(params_dims.size());
  for (auto d : params_dims) {
    auto curr_shape_type = mlir::LLVM::LLVMType::getArrayTy(llvm_I64_type, d);
    auto curr_type = mlir::LLVM::LLVMType::getStructTy(llvm_dialect,
        {llvm_I32_ptr_type, llvm_I32_ptr_type, llvm_I64_type, curr_shape_type, curr_shape_type}).getPointerTo();
    args_types.push_back(curr_type);
  }

  auto llvm_fn_type = mlir::LLVM::LLVMType::getFunctionTy(
      llvm_I32_type,
      mlir::ArrayRef<mlir::LLVM::LLVMType>(args_types
          /*{llvm_I32_type, llvm_I32_type//int32x1StructPtrType, int32x1StructPtrType}*/),
      /*isVarArg=*/false);

  mlir::PatternRewriter::InsertionGuard insert_guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<mlir::LLVM::LLVMFuncOp>(
      module.getLoc(),
      func_name,
      llvm_fn_type);
  return mlir::SymbolRefAttr::get(func_name, context);
}

// Insert the external function symbol to module
// pass base type params to the external function
//
mlir::FlatSymbolRefAttr GetOrInsertExternalFunc(
    mlir::PatternRewriter &rewriter,
    mlir::ModuleOp module,
    mlir::LLVM::LLVMDialect *llvm_dialect,
    const std::string& func_name) {
  auto *context = module.getContext();
  if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(func_name)) {
    return mlir::SymbolRefAttr::get(func_name, context);
  }
 
  auto llvm_I64_type = mlir::LLVM::LLVMType::getInt64Ty(llvm_dialect);
  auto llvm_I32_ptr_type = mlir::LLVM::LLVMType::getInt32Ty(llvm_dialect).getPointerTo();
  auto llvm_I32_type = mlir::LLVM::LLVMType::getInt32Ty(llvm_dialect);

  auto llvm_fn_type = mlir::LLVM::LLVMType::getFunctionTy(
      llvm_I32_type,
      mlir::ArrayRef<mlir::LLVM::LLVMType>({llvm_I32_type, llvm_I32_type}),
      /*isVarArg=*/false);

  mlir::PatternRewriter::InsertionGuard insert_guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<mlir::LLVM::LLVMFuncOp>(
      module.getLoc(),
      func_name,
      llvm_fn_type);
  return mlir::SymbolRefAttr::get(func_name, context);
}

} // namespace

// ------------------------------------------------------------
// Call3dExternalFuncOpLowering: _global_mlir_call_external_func_3d

mlir::PatternMatchResult
Call3dExternalFuncOpLowering::matchAndRewrite(mlir::Operation *op,
    llvm::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {

  auto loc = op->getLoc();
  auto *llvm_dialect =
      op->getContext()->getRegisteredDialect<mlir::LLVM::LLVMDialect>();
  assert(llvm_dialect && "expected llvm dialect to be registered");

  // Prepare types
  //
  auto get_void_ptr_type = mlir::LLVM::LLVMType::getInt8PtrTy(llvm_dialect);
  auto llvm_I32_type = mlir::LLVM::LLVMType::getInt32Ty(llvm_dialect);
  auto llvm_I64_type = mlir::LLVM::LLVMType::getInt64Ty(llvm_dialect);

  // Attention: multi params！
  // Load each param to args[i] 
  //
  std::vector<int64_t> params_dims;
  std::vector<mlir::SmallVector<mlir::Value, 4>> args;
  mlir::SmallVector<mlir::Type, 8> operand_types
      = {op->operand_type_begin(), op->operand_type_end()};
  args.resize(operand_types.size());
  int i = -1;
  // for loop: represent the count of params
  // 
  for (auto t : operand_types) {
    ++i;
    auto mem_ref_type = t.cast<mlir::MemRefType>();
    auto mem_ref_shape = mem_ref_type.getShape();
    int idx = 0;
    // for loop: represent the shape of current params
    //
    // push back current param's rank
    params_dims.push_back(mem_ref_shape.size());
    for (int64_t s : mem_ref_shape) {
      // TODO: should be Index here
      args[i].push_back(rewriter.create<mlir::LLVM::ConstantOp>(
                        loc, llvm_I64_type,
                        rewriter.getIntegerAttr(rewriter.getIndexType(), idx++)));
    }
  }

  // Load param
  // Func has two params: x and y
  //
  auto external_op = mlir::cast<mlir::TF::CallExternalFunc2Op>(op);

  // Get or insert external function to the parent module
  //
  mlir::ModuleOp parent_module = op->getParentOfType<mlir::ModuleOp>();
  auto func_ref = GetOrInsertMemrefExternalFunc(rewriter, parent_module,
      llvm_dialect, params_dims,
      "_global_mlir_call_external_func_3d");

  // Insert callOp to call the external function
  //
  rewriter.create<mlir::CallOp>(
      loc, func_ref, rewriter.getIntegerType(32),
      mlir::ArrayRef<mlir::Value>({external_op.x(), external_op.y()}));

  // Removed origin op
  rewriter.eraseOp(op);

  return matchSuccess();
}

// ------------------------------------------------------------
// CallExternalFuncOpLowering: _global_mlir_call_external_func

mlir::PatternMatchResult
CallExternalFuncOpLowering::matchAndRewrite(mlir::Operation *op,
    llvm::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {

  auto loc = op->getLoc();
  auto *llvm_dialect =
      op->getContext()->getRegisteredDialect<mlir::LLVM::LLVMDialect>();
  assert(llvm_dialect && "expected llvm dialect to be registered");

  // Prepare types
  //
  auto get_void_ptr_type = mlir::LLVM::LLVMType::getInt8PtrTy(llvm_dialect);
  auto llvm_I32_type = mlir::LLVM::LLVMType::getInt32Ty(llvm_dialect);
  auto llvm_I64_type = mlir::LLVM::LLVMType::getInt64Ty(llvm_dialect);

  // Attention: multi params！
  // Load each param to args[i] 
  //
  std::vector<mlir::SmallVector<mlir::Value, 4>> args;
  mlir::SmallVector<mlir::Type, 8> operand_types
      = {op->operand_type_begin(), op->operand_type_end()};
  args.resize(operand_types.size());
  int i = -1;
  // for loop: represent the count of params
  // 
  for (auto t : operand_types) {
    ++i;
    auto mem_ref_type = t.cast<mlir::MemRefType>();
    auto mem_ref_shape = mem_ref_type.getShape();
    int idx = 0;
    // for loop: represent the shape of current params
    //
    for (int64_t s : mem_ref_shape) {
      // TODO: should be Index here
      args[i].push_back(rewriter.create<mlir::LLVM::ConstantOp>(
                        loc, llvm_I64_type,
                        rewriter.getIntegerAttr(rewriter.getIndexType(), idx++)));
    }
  }

  // Load param
  // Func has two params: x and y
  //
  auto external_op = mlir::cast<mlir::TF::CallExternalFuncOp>(op);
  auto element_load0 = rewriter.create<mlir::LoadOp>(loc, external_op.x(), args[0]);
  auto element_load1 = rewriter.create<mlir::LoadOp>(loc, external_op.y(), args[1]);

  // Get or insert external function to the parent module
  //
  mlir::ModuleOp parent_module = op->getParentOfType<mlir::ModuleOp>();
  auto func_ref = GetOrInsertExternalFunc(rewriter, parent_module,
      llvm_dialect,
      "_global_mlir_call_external_func");

  // Insert callOp to call the external function
  //
  rewriter.create<mlir::CallOp>(
      loc, func_ref, rewriter.getIntegerType(32),
      mlir::ArrayRef<mlir::Value>({element_load0, element_load1/*, element_result*/}));

  // Removed origin op
  rewriter.eraseOp(op);

  /*
  // TODO: Vary params pattern
  //
  auto loc = op->getLoc();
  mlir::SmallVector<mlir::Type, 8> operand_types
      = {op->operand_type_begin(), op->operand_type_end()};
  std::vector<mlir::SmallVector<mlir::Value *, 4>> loopIvs;
  loopIvs.resize(operand_types.size());
  int i = -1;
  for (auto xxxt : operand_types) {
    ++i;
    auto mem_ref_type = xxxt.cast<mlir::MemRefType>();
    auto mem_ref_shape = mem_ref_type.getShape();
 
    for (unsigned i = 0, e = mem_ref_shape.size(); i != e; ++i) {
      // low index
      auto lowerBound = rewriter.create<mlir::ConstantIndexOp>(loc, 0);
      // high index
      auto upperBound = rewriter.create<mlir::ConstantIndexOp>(loc, mem_ref_shape[i]);
      auto step = rewriter.create<mlir::ConstantIndexOp>(loc, 1);
      auto loop =
          rewriter.create<mlir::loop::ForOp>(loc, lowerBound, upperBound, step);
      loop.getBody()->clear();
      loopIvs[i].push_back(loop.getInductionVar());
      rewriter.setInsertionPointToStart(loop.getBody());
      rewriter.create<mlir::loop::TerminatorOp>(loc);
      rewriter.setInsertionPointToStart(loop.getBody());
    }
  }

  auto external_op = mlir::cast<mlir::TF::CallExternalFuncOp>(op);
  auto element_load0 = rewriter.create<mlir::LoadOp>(loc, external_op.x(), loopIvs[0]);
  auto element_load1 = rewriter.create<mlir::LoadOp>(loc, external_op.y(), loopIvs[1]);
  */

  return matchSuccess();
}

// Return a symbol reference to the external function
mlir::FlatSymbolRefAttr
CallExternalFuncOpLowering::InternalGetOrInsertExternalFunc(
    mlir::PatternRewriter &rewriter,
    mlir::ModuleOp module,
    mlir::LLVM::LLVMDialect *llvm_dialect) {
  auto *context = module.getContext();
  if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("tensorflow::GlobalMlirCallExternalFunc")) {
    return mlir::SymbolRefAttr::get("tensorflow::GlobalMlirCallExternalFunc", context);
  }

  // Create a function declaration
  auto llvm_I32_type = mlir::LLVM::LLVMType::getInt32Ty(llvm_dialect);
  auto llvm_I8_ptr_type = mlir::LLVM::LLVMType::getInt8PtrTy(llvm_dialect);
  auto llvm_fn_type = mlir::LLVM::LLVMType::getFunctionTy(llvm_I32_type, llvm_I8_ptr_type,
                                                          false); // TODO: isvarArgs, like printf ?

  // insert function into the module.
  mlir::PatternRewriter::InsertionGuard insert_guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(),
      "tensorflow::GlobalMlirCallExternalFunc", llvm_fn_type);
  return mlir::SymbolRefAttr::get("tensorflow::GlobalMlirCallExternalFunc", context);
}

} // namespace mlir
