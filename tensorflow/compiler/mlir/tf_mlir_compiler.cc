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

#include "tensorflow/compiler/mlir/tf_mlir_compiler.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate_cl.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_lower_pass.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/runtime/dynamic_memref.h"
#include "tensorflow/core/framework/tensor.h"


#include "absl/strings/str_split.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassNameParser.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/TargetSelect.h"

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"  // TF:local_config_mlir
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"  // TF:local_config_mlir
#include "mlir/Transforms/Passes.h"  // TF:local_config_mlir
#include "mlir/Pass/PassManager.h"  // TF:local_config_mlir
#include "mlir/Pass/PassManager.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "mlir/Support/TranslateClParser.h"  // TF:local_config_mlir
#include "mlir/ExecutionEngine/OptUtils.h" // TF:local_config_mlir

namespace tensorflow {

const char* kPythonLibPath = "TF_PYTHON_LIB_PATH";
SimpleMlirCompiler::SimpleMlirCompiler(const std::string& graph_str): graph_stref_(graph_str), mlir_context_(){
  
  mlir_module_ = tensorflow::GraphdefToMlirTranslateFunction(
      graph_str, debug_info_file, input_arrays, input_dtypes, input_shapes,
      output_arrays, control_output_arrays, prune_unused_nodes,
      convert_legacy_fed_inputs, true, upgrade_legacy, &mlir_context_);
  mlir::registerPassManagerCLOptions();
}

Status SimpleMlirCompiler::CompileGraphDef(bool enableOpt)   {
  /*
  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  mlir::SourceMgrDiagnosticHandler diagnostic_handler(source_mgr, &context);
*/

  // Step1: generate mlir llvm dialect through graphdef
  mlir::PassManager pm(&mlir_context_);
  // Apply any generic pass manager command line options and run the pipeline.
  applyPassManagerCLOptions(pm);
  // -tf-executor-to-tf-conversion --xla-legalize-tf -hlo-legalize-to-lhlo --lhlo-legalize-to-linalg -lhlo-legalize-to-std --convert-linalg-to-affine-loops  --lower-affine  --convert-loop-to-std -convert-tf-to-llvm --convert-std-to-llvm
  pm.addPass(mlir::CreateTFExecutorToTFDialectConversion());
  pm.addPass(mlir::xla_hlo::createLegalizeTFPass());
  pm.addPass(mlir::xla_hlo::createLegalizeToLhloPass());
  pm.addPass(mlir::xla_lhlo::createLegalizeLhloToLinalgPass());
  pm.addPass(mlir::xla_lhlo::createLegalizeToStdPass());

  pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createLowerToCFGPass());
  pm.addPass(mlir::CreateTFLowerToLLVMPass());
  pm.addPass(mlir::createLowerToLLVMPass());


  if (mlir::failed(pm.run(*mlir_module_)))
    return errors::Internal("pass run failed");

  // mlir_module_->dump();

  // Step2: use mlir llvm dialect to generate function pointer
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Failed to create a JITTargetMachineBuilder for the host\n";
    return errors::Internal("Failed to create a JITTargetMachineBuilder for the host");
  }
  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Failed to create a TargetMachine for the host\n";
    return errors::Internal("Failed to create a TargetMachine for the host");
  }
  // An optimization pipeline to use within the execution engine.
  auto transformer = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/tmOrError->get());

  std::vector<std::string> clSharedLibs;
  //clSharedLibs.push_back("/home/tops/lib/libpython3.7m.so.1.0");
  //clSharedLibs.push_back("/home/tops/lib/python3.7/site-packages/tensorflow_core/libtensorflow_framework.so.2");
  const char* python_lib_path= getenv(kPythonLibPath);
  if (python_lib_path) {
    std::string python_libpath_str = std::string(python_lib_path);
    clSharedLibs.push_back(python_libpath_str + "/tensorflow_core/python/_pywrap_tensorflow_internal.so");
  } else {
    // use tops python3 as default
    clSharedLibs.push_back("/home/tops/lib/python3.7/site-packages/tensorflow_core/python/_pywrap_tensorflow_internal.so");
  }



  llvm::Optional<llvm::CodeGenOpt::Level> jitCodeGenOptLevel = static_cast<llvm::CodeGenOpt::Level>(3);
  llvm::SmallVector<llvm::StringRef, 4> libs(clSharedLibs.begin(), clSharedLibs.end());
  auto expectedEngine = mlir::ExecutionEngine::create(*mlir_module_, transformer,
      jitCodeGenOptLevel, libs);
  if (!expectedEngine) {
    llvm::errs() << "execution engine create error\n";
    return errors::Internal("execution engine create error");
  }

  auto engine = std::move(*expectedEngine);
  auto expectedFPtr = engine->lookup("main");
  if (!expectedFPtr) {
    llvm::errs() << "get function pointer error\n";
    return errors::Internal("get function pointer error");
  }
  engine_ = std::move(engine);

  return Status::OK();
}

Status SimpleMlirCompiler::RunJit(std::vector<void*>* args_pointers ) {

  auto expectedFPtr = engine_->lookup("main");
   void (*fptr)(void **) = *expectedFPtr;
  (*fptr)(((void**)(args_pointers->data())));


  return Status::OK();
}


}


