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

// This file implements logic for lowering LHLO dialect to Affine dialect.

#include "absl/memory/memory.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"  // TF:llvm-project
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/Transforms/DialectConversion.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/map_xla_to_scalar_op.h"

namespace mlir {
namespace xla_lhlo {
namespace {

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


struct UniqueCountConverter : public OpRewritePattern<UniqueCountOp> {
  using OpRewritePattern<UniqueCountOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(UniqueCountOp op,
                                     PatternRewriter& rewriter) const override {


    auto context = rewriter.getContext();
    auto loc = op.getLoc();
    std::vector<Type> input_types;
    std::vector<Type> result_types;
    input_types.push_back(op.input().getType());
    input_types.push_back(rewriter.getIndexType());
    result_types.push_back(op.input().getType().dyn_cast<MemRefType>().getElementType());
    FlatSymbolRefAttr output_func_ref = CallExternalFunc(
        rewriter, op.getParentOfType<ModuleOp>(), loc,
        "_global_get_unique_ids_count",
        input_types, result_types,
        ArrayRef<NamedAttribute>{});

    // Create a CallOp to call `_global_get_unique_ids_count`
    auto runtime_ids_count = rewriter.create<mlir::DimOp>(op.getLoc(), op.input(), 0);
    SmallVector<Value, 4> unique_ids_count_func_param;
    unique_ids_count_func_param.push_back(op.input());
    unique_ids_count_func_param.push_back(runtime_ids_count);
    // get unique ids count, params: ids memref + runtime ids count

    auto get_unique_count_op = rewriter.create<mlir::CallOp>(
        loc, output_func_ref.getValue(),
        result_types,
        unique_ids_count_func_param);

    rewriter.create<StoreOp>(op.getLoc(), get_unique_count_op.getResult(0), op.output());
    rewriter.eraseOp(op);
    return this->matchSuccess();
  }
 
};

struct UniqueIdsConverter : public OpRewritePattern<UniqueIdsOp> {
  using OpRewritePattern<UniqueIdsOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(UniqueIdsOp op,
                                     PatternRewriter& rewriter) const override {


    auto context = rewriter.getContext();
    auto loc = op.getLoc();
    //auto input_ids_type = op.input().getType().dyn_cast<MemRefType>().getElementType()

    std::vector<Type> input_types;
    std::vector<Type> result_types;
    input_types.push_back(op.lhs().getType());
    input_types.push_back(op.rhs().getType());
    input_types.push_back(op.out().getType());
    
    FlatSymbolRefAttr output_func_ref = CallExternalFunc(
        rewriter, op.getParentOfType<ModuleOp>(), loc,
        "_global_unique_ids",
        input_types, result_types,
        ArrayRef<NamedAttribute>{});

    // Create a CallOp to call `_global_get_unique_ids_count`
    SmallVector<Value, 4> func_param;
    func_param.push_back(op.lhs());
    func_param.push_back(op.rhs());
    func_param.push_back(op.out());

    auto unique_call = rewriter.create<mlir::CallOp>(
        loc, output_func_ref.getValue(),
        result_types,
        func_param);

    rewriter.eraseOp(op);
    return this->matchSuccess();
  }
};

struct UniqueIndexConverter : public OpRewritePattern<UniqueIndexOp> {
  using OpRewritePattern<UniqueIndexOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(UniqueIndexOp op,
                                     PatternRewriter& rewriter) const override {

    auto context = rewriter.getContext();
    auto loc = op.getLoc();
    auto output_type = op.out().getType().dyn_cast<MemRefType>().getElementType();
 
    std::vector<Type> input_types;
    std::vector<Type> result_types;
    input_types.push_back(op.lhs().getType());
    input_types.push_back(op.rhs().getType());
    input_types.push_back(op.out().getType());
    
    FlatSymbolRefAttr output_func_ref = CallExternalFunc(
        rewriter, op.getParentOfType<ModuleOp>(), loc,
        output_type.isInteger(64) ? "_global_unique_index64":"_global_unique_index32",
        input_types, result_types,
        ArrayRef<NamedAttribute>{});

    // Create a CallOp to call `_global_get_unique_ids_count`
    SmallVector<Value, 4> func_param;
    func_param.push_back(op.lhs());
    func_param.push_back(op.rhs());
    func_param.push_back(op.out());

    auto unique_call = rewriter.create<mlir::CallOp>(
        loc, output_func_ref.getValue(),
        result_types,
        func_param);

    rewriter.eraseOp(op);
    return this->matchSuccess();
  }
};


struct DebugPrintConverter : public OpRewritePattern<DebugPrintOp> {
  using OpRewritePattern<DebugPrintOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(DebugPrintOp op,
                                     PatternRewriter& rewriter) const override {


    auto loc = op.getLoc();
    ModuleOp parent_module = op.getParentOfType<ModuleOp>();
    auto context = rewriter.getContext();

    auto ele_type = op.input().getType().dyn_cast<MemRefType>();
    if (!ele_type)
      return this->matchFailure();
    
    std::string external_func_name =
      "_global_print_memref_" + std::to_string(ele_type.getRank()) + "d";

    std::vector<Type> input_types;
    std::vector<Type> result_types;
    input_types.push_back(ele_type);
  

    if (ele_type.getElementType().isInteger(64)) {
      external_func_name += "_i64";
    } else if (ele_type.getElementType().isInteger(32)) {
      // nothing
    } else if (ele_type.getElementType().isF64()) {
      external_func_name += "_f64";
    } else if (ele_type.getElementType().isF32()) {
      external_func_name += "_f32";
    } else if (ele_type.getElementType().isInteger(1)) {
      external_func_name += "_i1";
    } else {
      llvm::errs() << "Now only support int32 and int64 type.\n";
      return this->matchFailure();
    }

    auto print_tensor_ref = CallExternalFunc(
        rewriter, parent_module, loc,
        external_func_name,
        input_types, result_types,
        ArrayRef<NamedAttribute>{});

    // Create a CallOp to call `_global_print_memref_xd`
    SmallVector<Value, 4> func_params;
    func_params.push_back(op.input());
    rewriter.create<mlir::CallOp>(
        loc, print_tensor_ref.getValue(), result_types,
        func_params);

    rewriter.eraseOp(op);
    return this->matchSuccess();
  }
};

void populateLHLOToStdConversionPattern(MLIRContext* context,
                                           OwningRewritePatternList* patterns) {
  // clang-format off
  patterns->insert<
      UniqueCountConverter,
      UniqueIdsConverter,
      UniqueIndexConverter,
      DebugPrintConverter
    >(context);
  // clang-format on
}

struct LhloLegalizeToStd: public FunctionPass<LhloLegalizeToStd> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    ConversionTarget target(getContext());
    target.addLegalDialect<StandardOpsDialect>();
    // NOTE(jiankeng.pt): Advanced skills: prevent the error of UniqueOp lowering process.
    // Cause the `func` op here has no one legalize pattern.
    // If you don't want the Op with concrete information which you specify
    // in the anonymous function be lowered at the pass, please do this.
    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return  
        op.getName() == "_global_unique_index32" ||
        op.getName() == "_global_unique_index64" ||
        op.getName() == "_global_get_unique_ids_count" ||
        op.getName() == "_global_unique_ids" ||
        op.getName() == "_global_print_memref_64" ||
        op.getName() == "_global_print_memref_1d_i64" ||
        op.getName() == "_global_print_memref_1d_i64i32" ||
        op.getName() == "_global_print_memref_1d" ||
        op.getName() == "_global_print_memref_2d" ||
        op.getName() == "_global_print_memref_2d_i64" ||
        op.getName() == "_global_print_memref_2d_f64" ||
        op.getName() == "_global_print_memref_2d_f32" ||
        op.getName() == "_global_print_memref_2d_i1"; });
    auto func = getFunction();
    populateLHLOToStdConversionPattern(func.getContext(), &patterns);
    if (failed(applyPartialConversion(func, target, patterns, nullptr))) {
      signalPassFailure();
    }
    // applyPatternsGreedily(func, patterns);
  }
};

}  // namespace

std::unique_ptr<OpPassBase<FuncOp>> createLegalizeToStdPass() {
  return absl::make_unique<LhloLegalizeToStd>();
}

static PassRegistration<LhloLegalizeToStd> legalize_pass(
    "lhlo-legalize-to-std", "Legalize from LHLO dialect to stdandard dialect");

}  // namespace xla_lhlo
}  // namespace mlir
