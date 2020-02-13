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

static const std::string c_lib_memcpy = "memcpy";
static const std::string llvm_memcpy = "llvm.memcpy.p0i8.p0i8.i64";

// Insert the external function symbol to module
// pass memref type params to the external function
//
mlir::FlatSymbolRefAttr GetOrInsertMemrefExternalFunc(
    mlir::PatternRewriter &rewriter,
    mlir::ModuleOp module,
    mlir::LLVM::LLVMDialect *llvm_dialect,
    const std::vector<int64_t>& params_dims,
    Type element_type,
    const std::string& func_name) {
  std::string real_func_name = func_name;
  if (element_type.isInteger(64)) {
    real_func_name += "_i64";
  } else if (element_type.isInteger(32)) {
    // nothing
  } else {
    llvm::dbgs() << "GetOrInsertMemrefExternalFunc: error element type, must be i64 or i32\n";
    return nullptr;
  }

  auto *context = module.getContext();
  if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(real_func_name)) {
    return mlir::SymbolRefAttr::get(real_func_name, context);
  }
 
  auto llvm_I64_type = mlir::LLVM::LLVMType::getInt64Ty(llvm_dialect);
  auto llvm_I64_ptr_type = mlir::LLVM::LLVMType::getInt64Ty(llvm_dialect).getPointerTo();
  auto llvm_I32_ptr_type = mlir::LLVM::LLVMType::getInt32Ty(llvm_dialect).getPointerTo();
  auto llvm_I32_type = mlir::LLVM::LLVMType::getInt32Ty(llvm_dialect);
  auto llvm_array_1xi64_type = mlir::LLVM::LLVMType::getArrayTy(llvm_I64_type, 1);
  //auto int32x1StructPtrType = mlir::LLVM::LLVMType::getStructTy(llvm_dialect,
  //    {llvm_I32_ptr_type, llvm_I32_ptr_type, llvm_I64_type, llvm_array_1xi64_type, llvm_array_1xi64_type}).getPointerTo();
  SmallVector<LLVM::LLVMType, 4> args_types;
  args_types.reserve(params_dims.size());
  // params_dims.size() id the Func input params' count
  for (auto d : params_dims) {
    auto curr_shape_type = mlir::LLVM::LLVMType::getArrayTy(llvm_I64_type, d);
    if (element_type.isInteger(32)) {
      auto curr_type = mlir::LLVM::LLVMType::getStructTy(llvm_dialect,
          {llvm_I32_ptr_type, llvm_I32_ptr_type, llvm_I64_type, curr_shape_type, curr_shape_type}).getPointerTo();
      args_types.push_back(curr_type);
    } else if (element_type.isInteger(64)) {
      auto curr_type = mlir::LLVM::LLVMType::getStructTy(llvm_dialect,
          {llvm_I64_ptr_type, llvm_I64_ptr_type, llvm_I64_type, curr_shape_type, curr_shape_type}).getPointerTo();
      args_types.push_back(curr_type);
    } else {
      llvm::dbgs() << "GetOrInsertMemrefExternalFunc: error element type, must be i64 or i32\n";
    }
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
      real_func_name,
      llvm_fn_type);
  return mlir::SymbolRefAttr::get(real_func_name, context);
}

// Insert the external function symbol to module
// pass base type params to the external function
//
mlir::FlatSymbolRefAttr GetOrInsertExternalFunc(
    mlir::PatternRewriter &rewriter,
    mlir::ModuleOp module,
    mlir::LLVM::LLVMDialect *llvm_dialect,
    const std::string& func_name,
    LLVM::LLVMType type) {
  auto *context = module.getContext();
  if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(func_name)) {
    return mlir::SymbolRefAttr::get(func_name, context);
  }
 
  auto llvm_I64_type = mlir::LLVM::LLVMType::getInt64Ty(llvm_dialect);
  auto llvm_I32_ptr_type = mlir::LLVM::LLVMType::getInt32Ty(llvm_dialect).getPointerTo();
  auto llvm_I32_type = mlir::LLVM::LLVMType::getInt32Ty(llvm_dialect);

  auto llvm_fn_type = mlir::LLVM::LLVMType::getFunctionTy(
      /*llvm_I32_type,*/type,
      mlir::ArrayRef<mlir::LLVM::LLVMType>(/*{llvm_I32_type, llvm_I32_type}*/{type, type}),
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
// CallNdExternalFuncOpLowering: _global_mlir_call_external_func_3d

mlir::PatternMatchResult
CallNdExternalFuncOpLowering::matchAndRewrite(mlir::Operation *op,
    llvm::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {

  // get the dims of value, for example:
  // dims=2, call the `_global_mlir_call_external_func_2d` function
  // dims=3, call the `_global_mlir_call_external_func_3d` function
  auto value_dims = op->getAttr("value_dims");
  auto value_dims_attr = value_dims.dyn_cast<mlir::ElementsAttr>().getValue(0); 
  auto value_dims_int = value_dims_attr.dyn_cast<mlir::IntegerAttr>().getInt();
  std::string external_func_name =
      "_global_mlir_call_external_func_" + std::to_string(value_dims_int) + "d";

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
  Type ele_type = operand_types[0].dyn_cast<MemRefType>().getElementType();
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
  auto external_op = mlir::cast<mlir::TF::RawDebugPrint2Op>(op);

  // Get or insert external function to the parent module
  //
  mlir::ModuleOp parent_module = op->getParentOfType<mlir::ModuleOp>();
  auto func_ref = GetOrInsertMemrefExternalFunc(rewriter, parent_module,
      llvm_dialect, params_dims, ele_type,
      external_func_name);

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
// RawDebugPrintOpLowering: _global_mlir_call_external_func

mlir::PatternMatchResult
RawDebugPrintOpLowering::matchAndRewrite(mlir::Operation *op,
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
  auto external_op = mlir::cast<mlir::TF::RawDebugPrintOp>(op);
  auto element_load0 = rewriter.create<mlir::LoadOp>(loc, external_op.x(), args[0]);
  auto element_load1 = rewriter.create<mlir::LoadOp>(loc, external_op.y(), args[1]);

  // Get or insert external function to the parent module
  //
  mlir::ModuleOp parent_module = op->getParentOfType<mlir::ModuleOp>();
  auto func_ref = GetOrInsertExternalFunc(rewriter, parent_module,
      llvm_dialect,
      "_global_mlir_call_external_func",
      llvm_I32_type);

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

  auto external_op = mlir::cast<mlir::TF::RawDebugPrintOp>(op);
  auto element_load0 = rewriter.create<mlir::LoadOp>(loc, external_op.x(), loopIvs[0]);
  auto element_load1 = rewriter.create<mlir::LoadOp>(loc, external_op.y(), loopIvs[1]);
  */

  return matchSuccess();
}

// -----------------------------------------------------------
// MemcpyOp
//
mlir::PatternMatchResult
MemcpyOpLowering::matchAndRewrite(
    Operation *op, llvm::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {

  auto *context = op->getContext();
  auto loc = op->getLoc();
  auto *llvm_dialect =
      op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
  assert(llvm_dialect && "expected llvm dialect to be registered");

  // Get a symbol reference to the memcpy function, inserting it if necessary.
  // TODO: refine to `GetOrInsertExternalFunc`
  ModuleOp parent_module = op->getParentOfType<ModuleOp>();
  auto memcpy_ref = GetOrInsertMemcpy(rewriter, parent_module, llvm_dialect);

  // dest memref
  Type dst_type = operands[0].getType().cast<LLVM::LLVMType>().getStructElementType(1);
  Value aligned_dst_mem = rewriter.create<LLVM::ExtractValueOp>(
      loc, dst_type, operands[0], rewriter.getI64ArrayAttr(1));
  Value aligned_i8_ptr_dst_mem = rewriter.create<LLVM::BitcastOp>(
      loc, LLVM::LLVMType::getInt8PtrTy(llvm_dialect), aligned_dst_mem);

  // src memref
  Type src_type = operands[1].getType().cast<LLVM::LLVMType>().getStructElementType(1);
  Value aligned_src_mem = rewriter.create<LLVM::ExtractValueOp>(
      loc, src_type, operands[1], rewriter.getI64ArrayAttr(1));
  Value aligned_i8_ptr_src_mem = rewriter.create<LLVM::BitcastOp>(
      loc, LLVM::LLVMType::getInt8PtrTy(llvm_dialect), aligned_src_mem);

  // copy data size
  Value i64_size = rewriter.create<LLVM::SExtOp>(
      loc, LLVM::LLVMType::getInt64Ty(llvm_dialect), operands[2]);

  // create a callop to call 'llvm memcpy'
  rewriter.create<CallOp>(
      loc, memcpy_ref, LLVM::LLVMType::getInt8PtrTy(llvm_dialect),
      ArrayRef<Value>(
          {aligned_i8_ptr_dst_mem, aligned_i8_ptr_src_mem, i64_size}));

  rewriter.eraseOp(op);

  return matchSuccess();
}

mlir::FlatSymbolRefAttr
MemcpyOpLowering::GetOrInsertMemcpy(
    ConversionPatternRewriter &rewriter,
    ModuleOp module,
    LLVM::LLVMDialect *llvm_dialect) {
  auto *context = module.getContext();

  if (module.lookupSymbol<LLVM::LLVMFuncOp>(c_lib_memcpy)) {
    return SymbolRefAttr::get(c_lib_memcpy, context);
  }

  auto llvm_void_type = LLVM::LLVMType::getVoidTy(llvm_dialect);
  auto llvm_i8_ptr_type = LLVM::LLVMType::getInt8PtrTy(llvm_dialect);
  auto llvm_i64_type = LLVM::LLVMType::getInt64Ty(llvm_dialect);
  auto llvm_fn_type = LLVM::LLVMType::getFunctionTy(
      llvm_i8_ptr_type,//llvm_void_type,
      ArrayRef<mlir::LLVM::LLVMType>({llvm_i8_ptr_type, llvm_i8_ptr_type, llvm_i64_type}),
      false);

  // Insert the memcpy function into the body of the parent module.
  PatternRewriter::InsertionGuard insert_guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(),
                                    c_lib_memcpy, llvm_fn_type);
  return SymbolRefAttr::get(c_lib_memcpy, context);
}

// ---------------------------------------------------------------------
// CopyResultOp
//
namespace {

// first param's type is "1xi64"
mlir::FlatSymbolRefAttr GetOrInsertCopyResultFunc(
    mlir::PatternRewriter &rewriter,
    mlir::ModuleOp module,
    mlir::LLVM::LLVMDialect *llvm_dialect,
    const std::vector<int64_t>& params_dims,
    Type src_ele_type,
    const std::string& func_name) {
  auto *context = module.getContext();
  if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(func_name)) {
    return mlir::SymbolRefAttr::get(func_name, context);
  }

  auto llvm_I64_type = mlir::LLVM::LLVMType::getInt64Ty(llvm_dialect);
  auto llvm_I64_ptr_type = mlir::LLVM::LLVMType::getInt64Ty(llvm_dialect).getPointerTo();
  auto llvm_I32_ptr_type = mlir::LLVM::LLVMType::getInt32Ty(llvm_dialect).getPointerTo();
  auto llvm_I32_type = mlir::LLVM::LLVMType::getInt32Ty(llvm_dialect);
  auto llvm_F64_ptr_type = mlir::LLVM::LLVMType::getDoubleTy(llvm_dialect).getPointerTo();
  auto llvm_F32_ptr_type = mlir::LLVM::LLVMType::getFloatTy(llvm_dialect).getPointerTo();

  SmallVector<LLVM::LLVMType, 4> args_types;
  args_types.reserve(params_dims.size());

  // first param: "1xi64"
  auto dest_shape_type = mlir::LLVM::LLVMType::getArrayTy(llvm_I64_type, params_dims[0]);
  auto dest_type = mlir::LLVM::LLVMType::getStructTy(llvm_dialect,
    {llvm_I64_ptr_type, llvm_I64_ptr_type, llvm_I64_type, dest_shape_type, dest_shape_type}).getPointerTo();
  args_types.push_back(dest_type);

  // second param: "?x?x...i32", "?x?x...i64", "?x?x...f32", "?x?x...f64"
  auto src_shape_type = mlir::LLVM::LLVMType::getArrayTy(llvm_I64_type, params_dims[1]);
  if (src_ele_type.isInteger(32)) {
    auto src_type = mlir::LLVM::LLVMType::getStructTy(llvm_dialect,
      {llvm_I32_ptr_type, llvm_I32_ptr_type, llvm_I64_type, src_shape_type, src_shape_type}).getPointerTo();
    args_types.push_back(src_type);
  } else if (src_ele_type.isInteger(64)) {
    auto src_type = mlir::LLVM::LLVMType::getStructTy(llvm_dialect,
      {llvm_I64_ptr_type, llvm_I64_ptr_type, llvm_I64_type, src_shape_type, src_shape_type}).getPointerTo();
    args_types.push_back(src_type); 
  } else if (src_ele_type.isF32()) {
    auto src_type = mlir::LLVM::LLVMType::getStructTy(llvm_dialect,
      {llvm_F32_ptr_type, llvm_F32_ptr_type, llvm_I64_type, src_shape_type, src_shape_type}).getPointerTo();
    args_types.push_back(src_type);
  } else if (src_ele_type.isF64()) {
    auto src_type = mlir::LLVM::LLVMType::getStructTy(llvm_dialect,
      {llvm_F64_ptr_type, llvm_F64_ptr_type, llvm_I64_type, src_shape_type, src_shape_type}).getPointerTo();
    args_types.push_back(src_type);
  } else {
    assert(false && "GetOrInsertCopyResultFunc: error element type, must be i32, i64, f32, f64");
  }
 
  auto llvm_fn_type = mlir::LLVM::LLVMType::getFunctionTy(
      llvm_I64_type,
      mlir::ArrayRef<mlir::LLVM::LLVMType>(args_types),
      false);

  mlir::PatternRewriter::InsertionGuard insert_guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<mlir::LLVM::LLVMFuncOp>(
      module.getLoc(),
      func_name,
      llvm_fn_type);

  return mlir::SymbolRefAttr::get(func_name, context);
}
 
}

mlir::PatternMatchResult
CopyResultOpLowering::matchAndRewrite(
    Operation *op, llvm::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto loc = op->getLoc();
  auto *llvm_dialect =
    op->getContext()->getRegisteredDialect<mlir::LLVM::LLVMDialect>();
  assert(llvm_dialect && "expected llvm dialect to be registered");

  std::string func_name("_global_set_external_memref_r0");
  mlir::SmallVector<mlir::Type, 8> operand_types
      = {op->operand_type_begin(), op->operand_type_end()};
  if (operand_types.size() != 2) {
    assert(false && "tf.CopyResult should be passed two params.");
    return matchFailure();
  }

  int src_rank = operand_types[1].dyn_cast<MemRefType>().getShape().size();
  switch (src_rank) {
    case 0: break;
    case 1: func_name = "_global_set_external_memref_r1"; break;
    case 2: func_name = "_global_set_external_memref_r2"; break;
    case 3: func_name = "_global_set_external_memref_r3"; break;
    case 4: func_name = "_global_set_external_memref_r4"; break;
    case 5: func_name = "_global_set_external_memref_r5"; break;
    default:
      assert(false && "Now only support from rank-0 to rank-5.");
      return matchFailure();
  }

  Type src_type = operand_types[1].dyn_cast<MemRefType>().getElementType();
  if (src_type.isInteger(32)) {
    func_name += "_i32";
  } else if (src_type.isInteger(64)) {
    func_name += "_i64";
  } else if (src_type.isF32()) {
    func_name += "_f32";
  } else if (src_type.isF64()) {
    func_name += "_f64";
  } else {
    llvm::dbgs() << "Now only support i32, i64, f32, f64 type.\n";
    return matchFailure();
  }

  std::vector<int64_t> params_dims;
  int i = -1;
  for (auto t : operand_types) {
    ++i;
    auto mem_ref_type = t.cast<mlir::MemRefType>();
    auto mem_ref_shape = mem_ref_type.getShape();
    int idx = 0;
    // push back current param's rank
    params_dims.push_back(mem_ref_shape.size());
  }

  // CopyResult Op
  auto external_op = mlir::cast<mlir::TF::CopyResultOp>(op);

  // Get or insert external function to the parent module
  // dest type always be : tensor<1xi64> / memref<1xi64>
  mlir::ModuleOp parent_module = op->getParentOfType<mlir::ModuleOp>();
  auto func_ref = GetOrInsertCopyResultFunc(rewriter, parent_module,
      llvm_dialect, params_dims, src_type,
      func_name);

  // Insert callOp to call the external function
  auto llvm_I64_type = mlir::LLVM::LLVMType::getInt64Ty(llvm_dialect);
  rewriter.create<mlir::CallOp>(
      loc, func_ref, llvm_I64_type,
      mlir::ArrayRef<mlir::Value>({external_op.dest(), external_op.src()}));

  rewriter.eraseOp(op);

  return matchSuccess();
}

} // namespace mlir
