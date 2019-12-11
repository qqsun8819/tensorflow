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

} // end mlir
