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

#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_lower_to_affine.h"
#include "tensorflow/compiler/mlir/tensorflow/runtime/dynamic_memref.h"

#include "mlir/IR/AttributeSupport.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/Sequence.h"

namespace mlir {

namespace {

// Convert the given TensorType into the corresponding MemRefType.
//
static mlir::MemRefType ConvertTensorToMemRef(mlir::TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return mlir::MemRefType::get(type.getShape(), type.getElementType());
}

// Insert an allocation and deallocation for the given MemRefType.
//
static mlir::Value InsertAllocAndDealloc(
    mlir::MemRefType type, mlir::Location loc,
    mlir::PatternRewriter &rewriter) {
  auto alloc = rewriter.create<mlir::AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parent_block = alloc.getOperation()->getBlock();
  alloc.getOperation()->moveBefore(&parent_block->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as toy functions have no control flow.
  auto dealloc = rewriter.create<mlir::DeallocOp>(loc, alloc);
  dealloc.getOperation()->moveBefore(&parent_block->back());
  return alloc;
}

} // namespace

mlir::PatternMatchResult
ConstOpLowering::matchAndRewrite(
    mlir::TF::ConstOp op,
    mlir::PatternRewriter &rewriter) const {
  
  mlir::ElementsAttr constant_value = op.value();
  mlir::Location loc = op.getLoc();

  auto tensor_type = op.getType().cast<mlir::TensorType>();
  auto mem_ref_type = ConvertTensorToMemRef(tensor_type);
  auto alloc = InsertAllocAndDealloc(mem_ref_type, loc, rewriter);

  auto value_shape = mem_ref_type.getShape();

  mlir::SmallVector<mlir::Value, 8> constant_indices;
  // TODO: scalar element shape = 0
  if (value_shape.size() > 0) {
    for (auto i : llvm::seq<int64_t>(
             0, *std::max_element(value_shape.begin(), value_shape.end())))
      constant_indices.push_back(rewriter.create<mlir::ConstantIndexOp>(loc, i));
  }

  mlir::SmallVector<mlir::Value, 2> indices;
  // TODO: Float Attr ..
  auto value_it = constant_value.getValues<mlir::IntegerAttr>().begin();
  std::function<void(uint64_t)> store_elements = [&](uint64_t dimension) {
    if (dimension == value_shape.size()) {
      rewriter.create<mlir::AffineStoreOp>(
          loc, rewriter.create<mlir::ConstantOp>(loc, *value_it++), alloc,
          llvm::makeArrayRef(indices));
      return;
    }

    for (uint64_t i = 0, e = value_shape[dimension]; i != e; ++i) {
      indices.push_back(constant_indices[i]);
      store_elements(dimension + 1);
      indices.pop_back();
    }
  };

  store_elements(0);

  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, alloc);

  return matchSuccess();
}
FlatSymbolRefAttr CallExternalFunc(
    PatternRewriter &rewriter,
    ModuleOp module,
    Location loc,
    const std::string& func_name,
    std::vector<Type> &input_types,
    std::vector<Type> &result_types,
    ArrayRef<NamedAttribute> attrs) {

  // TODO: FIXME, how about the same function name, by differernt params ?
  // External function symbol is existed
  FlatSymbolRefAttr func_name_attr = rewriter.getSymbolRefAttr(func_name);
  if (module.lookupSymbol(func_name)) {
    return func_name_attr;
  }

  auto context = rewriter.getContext();
  // function type
  auto func_type = FunctionType::get(input_types, result_types, context);

  // create func op
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<mlir::FuncOp>(loc, func_name_attr.getValue(),
                                func_type, attrs);

  return func_name_attr;
}
// --------------------------------------------------------
// CopyResultOp
//
mlir::PatternMatchResult
CopyResultOpLowering::matchAndRewrite(
    mlir::TF::CopyResultOp op,
    mlir::PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  ModuleOp parent_module = op.getParentOfType<ModuleOp>();
  auto context = rewriter.getContext();

  std::vector<Type> result_types;
  std::vector<Type> input_types;

  auto src_memref_type = op.src().getType();
  auto dest_memref_type = op.dest().getType();
  auto src_ele_type = src_memref_type.dyn_cast<MemRefType>();
  auto dest_ele_type = dest_memref_type.dyn_cast<MemRefType>();

  if (!dest_ele_type) {
    auto dest_tensor_type = dest_memref_type.dyn_cast<RankedTensorType>();
    auto dest_real_memref = ConvertTensorToMemRef(dest_tensor_type);
    dest_ele_type = dest_real_memref;
    input_types.push_back(dest_real_memref);
  } else {
    input_types.push_back(dest_memref_type);
  }

  if (!src_ele_type) {
    auto src_tensor_type = src_memref_type.dyn_cast<RankedTensorType>();
    auto src_real_memref = ConvertTensorToMemRef(src_tensor_type);
    src_ele_type = src_real_memref;
    input_types.push_back(src_real_memref);
  } else {
    input_types.push_back(src_memref_type);
  }

  // TODO: unknow rank case ?
  //
  int rank = src_ele_type.getRank();
  std::string func_name("_global_set_external_memref_r0");
  switch (rank) {
    case 0: break;
    case 1: func_name = "_global_set_external_memref_r1"; break;
    case 2: func_name = "_global_set_external_memref_r2"; break;
    case 3: func_name = "_global_set_external_memref_r3"; break;
    case 4: func_name = "_global_set_external_memref_r4"; break;
    case 5: func_name = "_global_set_external_memref_r5"; break;
    default:
      llvm::dbgs() << "Now only support from rank-0 to rank-5.\n";
      return matchFailure();
  }

  //input_types.push_back(IntegerType::get(32, context));
  Value vtype = nullptr;
  if (src_ele_type.getElementType().isInteger(32)) {
    vtype = rewriter.create<ConstantOp>(
        loc, rewriter.getIntegerAttr(IntegerType::get(32, context), 0));
    func_name += "_i32";
  } else if (src_ele_type.getElementType().isInteger(64)) {
    vtype = rewriter.create<ConstantOp>(
        loc, rewriter.getIntegerAttr(IntegerType::get(32, context), 1));
    func_name += "_i64";
  } else if (src_ele_type.getElementType().isF32()) {
    vtype = rewriter.create<ConstantOp>(
        loc, rewriter.getIntegerAttr(IntegerType::get(32, context), 2));
    func_name += "_f32";
  } else if (src_ele_type.getElementType().isF64()) {
    vtype = rewriter.create<ConstantOp>(
        loc, rewriter.getIntegerAttr(IntegerType::get(32, context), 3));
    func_name += "_f64";
  } else {
    llvm::dbgs() << "Now only support i32, i64, f32, f64 type.\n";
    return matchFailure();
  }

  auto set_external_memref = CallExternalFunc(
      rewriter, parent_module, loc, func_name,
      input_types, result_types,
      ArrayRef<NamedAttribute>{});

  // Create a CallOp to call `_global_set_external_memref_rx`
  SmallVector<Value, 4> func_params;
  func_params.push_back(op.dest());
  func_params.push_back(op.src());
  //func_params.push_back(vtype);
  rewriter.create<mlir::CallOp>(
      loc, set_external_memref.getValue(), result_types,
      func_params);

  rewriter.eraseOp(op);

  return matchSuccess();
}


} // end mlir
