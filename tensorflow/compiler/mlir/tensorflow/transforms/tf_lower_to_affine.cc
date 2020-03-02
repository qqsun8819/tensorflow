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

// int64: return 8B
// int32: return 4B ...
unsigned GetMemRefElementSizeInBytes(MemRefType memref_type) {
  auto element_type = memref_type.getElementType();

  unsigned size_in_bits;
  if (element_type.isIntOrFloat()) {
    size_in_bits = element_type.getIntOrFloatBitWidth();
  } else {
    auto vector_type = element_type.cast<VectorType>();
    size_in_bits =
        vector_type.getElementTypeBitWidth() * vector_type.getNumElements();
  }
  return llvm::divideCeil(size_in_bits, 8); 
}

// All dimensions are known at compile time.
static bool AllDimensionsStatic(MemRefType type) {
  auto memref_shape = type.getShape();
  for (int i = 0; i < memref_shape.size(); ++i)
    if (memref_shape[i] < 0)
      return false;
  return true;
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

mlir::PatternMatchResult
ReshapeOpLOwering::matchAndRewrite(
    mlir::TF::ReshapeOp op,
    mlir::PatternRewriter &rewriter) const {

  // TODO: shape-rank value should not be dynamic shape
  // must be 1-D, like <3xi64>, <3xi32>
  auto shape_rank_type = op.shape().getType().dyn_cast<RankedTensorType>();
  if (!shape_rank_type) {
    return matchFailure();
  }

  // rank must equal 1
  if (shape_rank_type.getRank() != 1) {
    return matchFailure();
  }

  // not support dynamic shape-rank, must be <3xi64>, not <?xi64>
  int64_t shape_rank_dim_0 = shape_rank_type.getDimSize(0);
  if (TensorType::isDynamic(shape_rank_dim_0)) {
    return matchFailure();
  }
 
  auto loc = op.getLoc();
  auto result_tensor_type = op.output().getType().dyn_cast<TensorType>();
  if (!result_tensor_type) {
    return matchFailure();
  }
  auto result_memref_type = ConvertTensorToMemRef(result_tensor_type);
  auto result_shape = result_memref_type.getShape();
  if (result_shape.size() != shape_rank_dim_0) {
    llvm::dbgs() << "ReshapeOp wrong shape, expect rank=" << shape_rank_dim_0
                 << ", result rank=" << result_shape.size();
    return matchFailure();
  }
  Value output = nullptr;

  // Compute size in bytes, init to element-type size in Byte type
  // like int64, init total_tensor_size = 8B
  Value total_tensor_size = nullptr;
  unsigned element_size_in_byte = GetMemRefElementSizeInBytes(result_memref_type);

  if (AllDimensionsStatic(result_memref_type)) {
    // 1) Static shape
    llvm::dbgs() << "static reshape: " << result_memref_type << "\n";
    output = InsertAllocAndDealloc(result_memref_type, loc, rewriter);
    int64_t static_size = element_size_in_byte;
    for (auto s : result_memref_type.getShape()) {
      static_size *= s;
    }
    // static total tensor size
    total_tensor_size = rewriter.create<ConstantOp>(
        loc, rewriter.getIntegerAttr(
            rewriter.getIntegerType(64),
            static_size));

    // TODO: fix it
    // output = rewriter.create<MemRefCastOp>(loc, result_memref_type, op.getOperand(0));
  } else {
    // 2) Dynamic shape
    llvm::dbgs() << "dynamic reshape: " << result_memref_type << "\n";

    // init total tensor size
    total_tensor_size = rewriter.create<ConstantOp>(
        loc, rewriter.getIntegerAttr(
            rewriter.getIntegerType(64),
            element_size_in_byte));

    auto shape_ele_type = shape_rank_type.getElementType();
    bool is_shape_ele_type_i64 = shape_ele_type.isInteger(64);

    SmallVector<Value, 4> alloc_operands;
    // <3xi64> => shape[2x3x4]
    // shape_rank_dim_0 is static, so can be used in c++ for loop
    for (int i = 0; i < shape_rank_dim_0; ++i) {
      Value index = rewriter.create<ConstantOp>(
          loc, rewriter.getIntegerAttr(rewriter.getIndexType(), i));
      Value cur_dim_value = rewriter.create<LoadOp>(loc, op.getOperand(1), index);
      if (is_shape_ele_type_i64) {
        total_tensor_size = rewriter.create<MulIOp>(loc, total_tensor_size, cur_dim_value);
      } else {
        Value cur_dim_value_i64 = rewriter.create<ZeroExtendIOp>(
            loc, cur_dim_value, rewriter.getIntegerType(64));
        total_tensor_size = rewriter.create<MulIOp>(loc, total_tensor_size, cur_dim_value_i64);
      }

      // For example: <2x?xi64>, not all dims' value are -1
      if (result_shape[i] == -1) {
        alloc_operands.push_back(rewriter.create<IndexCastOp>(
            loc, cur_dim_value, rewriter.getIndexType()));
      } else {
        // reshape has static dim value at ith dim.
        // TODO: FIXME, should compare `result_shape[i]` and cur_dim_value,
        // if they are not equal, so error.
      }
    }

    // alloc output tensor-memref
    AllocOp allocate_memref =
        rewriter.create<AllocOp>(loc, result_memref_type, alloc_operands);
    auto *parent_block = allocate_memref.getOperation()->getBlock();
    // allocate_memref.getOperation()->moveBefore(&parent_block->front());
    // TODO: fix it, the time create Dealloc Op
    // dealloc
    auto dealloc = rewriter.create<DeallocOp>(loc, allocate_memref);
    dealloc.getOperation()->moveBefore(&parent_block->back());

    output = allocate_memref;
  }

  // copy(or store) data to new alloc
  // Here use a MemcpyOp
  rewriter.create<mlir::TF::MemcpyOp>(loc, output, op.getOperand(0), total_tensor_size);
  rewriter.replaceOp(op, output);

  return matchSuccess();
}

} // end mlir
