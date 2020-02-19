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
#include <unordered_set>


#include "llvm/Support/MemoryBuffer.h"
#include "mlir/Support/FileUtilities.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/compiler/mlir/tf_mlir_compiler.h"


static llvm::cl::opt<std::string> input_filename(llvm::cl::Positional,
                                                 llvm::cl::desc("<input file>"),
                                                 llvm::cl::init("-"));

int main(int argc, char** argv) {
  tensorflow::InitMlir y(&argc, &argv);

  llvm::cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  std::string error_message;

  llvm::outs() << "input file name:" << input_filename << " \n";
  auto input = mlir::openInputFile(input_filename, &error_message);

  if (!input) {
    llvm::errs() << error_message << "\n";
    return 1;
  }

  tensorflow::SimpleMlirCompiler mlir_compiler(input->getBuffer().str());
  auto s1 = mlir_compiler.CompileGraphDef();
  if (!s1.ok())  {
    LOG(ERROR) << "compile mlir faile";
    return -1;
  }
  llvm::outs() << "compile success\n";
  s1 = mlir_compiler.RunJit(true);
  if (!s1.ok())  {
    LOG(ERROR) << "run jit faile";
    return -1;
  }

  return 0;
}
