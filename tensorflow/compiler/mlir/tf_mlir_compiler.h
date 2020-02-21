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

#ifndef TENSORFLOW_COMPILER_MLIR_TF_MLIR_COMPILER_H_
#define TENSORFLOW_COMPILER_MLIR_TF_MLIR_COMPILER_H_

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Module.h"
#include "tensorflow/core/platform/status.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h" // TF:local_config_mlir
#include <memory>
namespace tensorflow {

class SimpleMlirCompiler {   
 public:
  SimpleMlirCompiler(const std::string& graph_str,
                     const std::string& entry_func_name);
  Status CompileGraphDef(bool enable_opt);
  Status RunJit(std::vector<void*>* args_pointers); 
 private:
  llvm::StringRef graph_stref_;
  mlir::MLIRContext mlir_context_;
  mlir::OwningModuleRef mlir_module_;
  std::unique_ptr<mlir::ExecutionEngine> engine_;
  std::string entry_func_name_;
};  
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TF_MLIR_COMPILER_H_
