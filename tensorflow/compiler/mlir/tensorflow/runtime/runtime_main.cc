/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// 1) bazel build --config=opt --config=noaws --config=nohdfs
//    --config=nonccl  //tensorflow/compiler/mlir/tensorflow:tf-mlir-runtime
//
// 2) RUN:
// bazel-bin/tensorflow/compiler/mlir/tensorflow/tf-mlir-runtime
//   -e main_1d_i32 | main_1d_i64 | main_2d_i32 | main_2d_i64 | main_2d_f32 | main_2d_f64
//   -entry-point-result=user_define
//   -shared-libs=/path/libtf_mlir_mlir_compiler_util.so

#include "tensorflow/compiler/mlir/tensorflow/runtime/dynamic_memref.h"
#include "tensorflow/core/framework/tensor.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassNameParser.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include <numeric>
#include <stdlib.h>
#include <iostream>

using namespace mlir;
using llvm::Error;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));
static llvm::cl::opt<std::string>
    mainFuncName("e", llvm::cl::desc("The function to be called"),
                 llvm::cl::value_desc("<function name>"),
                 llvm::cl::init("main"));
static llvm::cl::opt<std::string> mainFuncType(
    "entry-point-result",
    llvm::cl::desc("Textual description of the function type to be called"),
    llvm::cl::value_desc("f32 | void"), llvm::cl::init("f32"));

static llvm::cl::OptionCategory optFlags("opt-like flags");

// CLI list of pass information
static llvm::cl::list<const llvm::PassInfo *, bool, llvm::PassNameParser>
    llvmPasses(llvm::cl::desc("LLVM optimizing passes to run"),
               llvm::cl::cat(optFlags));

// CLI variables for -On options.
static llvm::cl::opt<bool>
    optO0("O0", llvm::cl::desc("Run opt passes and codegen at O0"),
          llvm::cl::cat(optFlags));
static llvm::cl::opt<bool>
    optO1("O1", llvm::cl::desc("Run opt passes and codegen at O1"),
          llvm::cl::cat(optFlags));
static llvm::cl::opt<bool>
    optO2("O2", llvm::cl::desc("Run opt passes and codegen at O2"),
          llvm::cl::cat(optFlags));
static llvm::cl::opt<bool>
    optO3("O3", llvm::cl::desc("Run opt passes and codegen at O3"),
          llvm::cl::cat(optFlags));

static llvm::cl::OptionCategory clOptionsCategory("linking options");
static llvm::cl::list<std::string>
    clSharedLibs("shared-libs", llvm::cl::desc("Libraries to link dynamically"),
                 llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated,
                 llvm::cl::cat(clOptionsCategory));

// CLI variables for debugging.
static llvm::cl::opt<bool> dumpObjectFile(
    "dump-object-file",
    llvm::cl::desc("Dump JITted-compiled object to file specified with "
                   "-object-filename (<input file>.o by default)."));

static llvm::cl::opt<std::string> objectFilename(
    "object-filename",
    llvm::cl::desc("Dump JITted-compiled object to file <input file>.o"));

static OwningModuleRef parseMLIRInput(StringRef inputFilename,
                                      MLIRContext *context) {
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  return OwningModuleRef(parseSourceFile(sourceMgr, context));
}

// Initialize the relevant subsystems of LLVM.
static void initializeLLVM() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
}

static inline Error make_string_error(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}

static llvm::Optional<unsigned> getCommandLineOptLevel() {
  llvm::Optional<unsigned> optLevel;
  llvm::SmallVector<std::reference_wrapper<llvm::cl::opt<bool>>, 4> optFlags{
      optO0, optO1, optO2, optO3};

  // Determine if there is an optimization flag present.
  for (unsigned j = 0; j < 4; ++j) {
    auto &flag = optFlags[j].get();
    if (flag) {
      optLevel = j;
      break;
    }
  }
  return optLevel;
}

// JIT-compile the given module and run "entryPoint" with "args" as arguments.
static Error
compileAndExecute(ModuleOp module, StringRef entryPoint,
                  std::function<llvm::Error(llvm::Module *)> transformer,
                  void **args) {
  Optional<llvm::CodeGenOpt::Level> jitCodeGenOptLevel;
  if (auto clOptLevel = getCommandLineOptLevel())
    jitCodeGenOptLevel =
        static_cast<llvm::CodeGenOpt::Level>(clOptLevel.getValue());
  SmallVector<StringRef, 4> libs(clSharedLibs.begin(), clSharedLibs.end());
  auto expectedEngine = mlir::ExecutionEngine::create(module, transformer,
                                                      jitCodeGenOptLevel, libs);
  if (!expectedEngine)
    return expectedEngine.takeError();

  auto engine = std::move(*expectedEngine);
  auto expectedFPtr = engine->lookup(entryPoint);
  if (!expectedFPtr)
    return expectedFPtr.takeError();

  if (dumpObjectFile)
    engine->dumpToObjectFile(objectFilename.empty() ? inputFilename + ".o"
                                                    : objectFilename);

  void (*fptr)(void **) = *expectedFPtr;
  (*fptr)(args);

  return Error::success();
}

static Error compileAndExecuteVoidFunction(
    ModuleOp module, StringRef entryPoint,
    std::function<llvm::Error(llvm::Module *)> transformer) {
  auto mainFunction = module.lookupSymbol<LLVM::LLVMFuncOp>(entryPoint);
  if (!mainFunction || mainFunction.getBlocks().empty())
    return make_string_error("entry point not found");
  void *empty = nullptr;
  return compileAndExecute(module, entryPoint, transformer, &empty);
}

// TODO: Return value example
//
static Error compileAndExecuteSingleFloatReturnFunction(
    ModuleOp module, StringRef entryPoint,
    std::function<llvm::Error(llvm::Module *)> transformer) {
  auto mainFunction = module.lookupSymbol<LLVM::LLVMFuncOp>(entryPoint);
  if (!mainFunction || mainFunction.isExternal())
    return make_string_error("entry point not found");

  if (mainFunction.getType().getFunctionNumParams() != 0)
    return make_string_error("function inputs not supported");

  if (!mainFunction.getType().getFunctionResultType().isFloatTy())
    return make_string_error("only single llvm.f32 function result supported");

  float res;
  struct {
    void *data;
  } data;
  data.data = &res;
  if (auto error =
          compileAndExecute(module, entryPoint, transformer, (void **)&data))
    return error;

  // Intentional printing of the output so we can test.
  llvm::outs() << res << '\n';

  return Error::success();
}

// Test cases:
// 1-D int64_t, 2-D f32, 2-D int32_t,
// 1-D int64_t tensor, 2-D f32 tensor, 2-D f64 tensor
//
static Error compileAndExecuteFunctionWithArgs(
    ModuleOp module, StringRef entryPoint,
    std::function<llvm::Error(llvm::Module *)> transformer) {
  auto mainFunction = module.lookupSymbol<LLVM::LLVMFuncOp>(entryPoint);
  if (!mainFunction || mainFunction.isExternal())
    return make_string_error("entry point not found");

  // 1) Test 1-D int64_t
  if (mainFuncName.getValue() == "main_1d_i64") {
    std::cout << "Now testing `1-D int64_t` case\n";
    int count = 10;
    int64_t* ptr = (int64_t*)malloc(sizeof(int64_t)*count);
    for (int i = 0; i < count; ++i) {
      *(ptr+i) = (int64_t)(i);
    }
    std::vector<int64_t> shape;
    shape.push_back(count);

    mlir::runtime::InputTensorWrapper<1> input_tensor(ptr, shape);
    void* args_pointer = input_tensor.GetArg();

    if (auto error =
        compileAndExecute(module, entryPoint, transformer, ((void**)&args_pointer)))
      return error;
  }

  // 2) Test 2-D f32
  if (mainFuncName.getValue() == "main_2d_f32") {
    std::cout << "Now testing `2-D f32` case\n";
    const int count1 = 10;
    const int count2 = 5;
    float ptr[count1][count2];
    for (int i = 0; i < count1; ++i) {
      for (int j = 0; j < count2; ++j) {
        ptr[i][j] = 1.1 * (i * 10 + j);
      }
    }
    std::vector<int64_t> shape;
    shape.push_back(count1);
    shape.push_back(count2);

    mlir::runtime::InputTensorWrapper<2> input_tensor(ptr, shape);
    void* args_pointer = input_tensor.GetArg();

    if (auto error =
        compileAndExecute(module, entryPoint, transformer, ((void**)&args_pointer)))
      return error;
  }

  // 3) Test 2-D int32_t
  if (mainFuncName.getValue() == "main_2d_i32") {
    std::cout << "Now testing `2-D int32_t` case\n";
    const int count1 = 10;
    const int count2 = 5;
    int32_t ptr[count1][count2];
    for (int i = 0; i < count1; ++i) {
      for (int j = 0; j < count2; ++j) {
        ptr[i][j] = i * 10 + j;
      }
    }
    std::vector<int64_t> shape;
    shape.push_back(count1);
    shape.push_back(count2);

    mlir::runtime::InputTensorWrapper<2> input_tensor(ptr, shape);
    void* args_pointer = input_tensor.GetArg();

    if (auto error =
        compileAndExecute(module, entryPoint, transformer, ((void**)&args_pointer)))
      return error;
  }

  // 4) Test 1-D int64_t tensor
  if (mainFuncName.getValue() == "main_1d_i64") {
    std::cout << "Now testing `1-D int64_t tensor` case\n";
    tensorflow::Tensor tensor(tensorflow::DT_INT64, tensorflow::TensorShape({8}));
    for (int i = 0; i < 8; i++) {
      tensor.vec<tensorflow::int64>()(i) = i;
    }

    mlir::runtime::InputTensorWrapper<1> input_tensor(tensor);
    void* args_pointer = input_tensor.GetArg();

    if (auto error =
        compileAndExecute(module, entryPoint, transformer, ((void**)&args_pointer)))
      return error;
  }

  // 5) Test 2-D f32 tensor
  if (mainFuncName.getValue() == "main_2d_f32") {
    std::cout << "Now testing `2-D f32 tensor` case\n";
    tensorflow::Tensor tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({8, 8}));
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        tensor.matrix<float>()(i, j) = 1.1 * (i * 10 + j);
      }
    }

    mlir::runtime::InputTensorWrapper<2> input_tensor(tensor);
    void* args_pointer = input_tensor.GetArg();

    if (auto error =
        compileAndExecute(module, entryPoint, transformer, ((void**)&args_pointer)))
      return error;
  }

  // 6) Test 2-D f64 tensor
  if (mainFuncName.getValue() == "main_2d_f64") {
    std::cout << "Now testing `2-D f64 tensor` case\n";
    tensorflow::Tensor tensor(tensorflow::DT_DOUBLE, tensorflow::TensorShape({5, 3}));
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 3; ++j) {
        tensor.matrix<double>()(i, j) = 1.1 * (i * 10 + j);
      }
    }

    mlir::runtime::InputTensorWrapper<2> input_tensor(tensor);
    void* args_pointer = input_tensor.GetArg();

    if (auto error =
        compileAndExecute(module, entryPoint, transformer, ((void**)&args_pointer)))
      return error;
  }

  // 7) Test input args && return results
  //
  if (mainFuncName.getValue() == "main_ret1") {
    std::cout << "Now testing `input args && return results` case\n";
    int count = 1;
    int64_t* ptr = (int64_t*)malloc(sizeof(int64_t)*count);
    for (int i = 0; i < count; ++i) {
      *(ptr+i) = 999;
    }
    std::vector<int64_t> shape;
    shape.push_back(count);
    std::vector<void*> args_pointers;

    mlir::runtime::InputTensorWrapper<1> input_tensor(ptr, shape);
    void* args_pointer = input_tensor.GetArg();
    args_pointers.push_back(args_pointer);

    // Create a ResultTensorWrapper for result tensor
    mlir::runtime::ResultTensorWrapper *rtw = new mlir::runtime::ResultTensorWrapper();
    args_pointers.push_back(rtw->GetArg());

    if (auto error =
        compileAndExecute(module, entryPoint, transformer, ((void**)(args_pointers.data()))))
      return error;

    tensorflow::Tensor* result_tensor_ptr = rtw->GetResultTensorPointer();
    LOG(INFO) << "result tensor = " << result_tensor_ptr->DebugString(128) << "\n";

    // TODO: set result tensor to output
    tensorflow::Tensor tensor_copied(*result_tensor_ptr);
    delete rtw;
    LOG(INFO) << "result tensor copied = " << tensor_copied.DebugString(128) << "\n";
  }

  // 8) Test input args && return results
  if (mainFuncName.getValue() == "main_ret2") {
    std::vector<void*> args_pointers;

    // Create a ResultTensorWrapper for result tensor
    mlir::runtime::ResultTensorWrapper *rtw = new mlir::runtime::ResultTensorWrapper();
    args_pointers.push_back(rtw->GetArg());

    if (auto error =
        compileAndExecute(module, entryPoint, transformer, ((void**)(args_pointers.data()))))
      return error;

    tensorflow::Tensor* result_tensor_ptr = rtw->GetResultTensorPointer();
    LOG(INFO) << "result tensor = " << result_tensor_ptr->DebugString(128) << "\n";

    // TODO: set result tensor to output
    tensorflow::Tensor tensor_copied(*result_tensor_ptr);
    delete rtw;
    LOG(INFO) << "result tensor copied = " << tensor_copied.DebugString(128) << "\n";
  }

  // 9) Test input args && return results
  if (mainFuncName.getValue() == "main_ret3") {
    std::vector<void*> args_pointers;

    const int count1 = 2;
    const int count2 = 3;
    double ptr0[count1][count2];
    double ptr1[count1][count2];
    for (int i = 0; i < count1; ++i) {
      for (int j = 0; j < count2; ++j) {
        ptr0[i][j] = 1.0;
        ptr1[i][j] = 2.0;
      }
    }

    std::vector<int64_t> shape;
    shape.push_back(count1);
    shape.push_back(count2);
 
    mlir::runtime::InputTensorWrapper<2> input_tensor0(ptr0, shape);
    void* args_pointer0 = input_tensor0.GetArg();
    args_pointers.push_back(args_pointer0);

    mlir::runtime::InputTensorWrapper<2> input_tensor1(ptr1, shape);
    void* args_pointer1 = input_tensor1.GetArg();
    args_pointers.push_back(args_pointer1);

    // Create a ResultTensorWrapper for result tensor
    mlir::runtime::ResultTensorWrapper *rtw = new mlir::runtime::ResultTensorWrapper();
    args_pointers.push_back(rtw->GetArg());

    if (auto error =
        compileAndExecute(module, entryPoint, transformer, ((void**)(args_pointers.data()))))
      return error;

    tensorflow::Tensor* result_tensor_ptr = rtw->GetResultTensorPointer();
    LOG(INFO) << "result tensor = " << result_tensor_ptr->DebugString(128) << "\n";

    // TODO: set result tensor to output
    tensorflow::Tensor tensor_copied(*result_tensor_ptr);
    delete rtw;
    LOG(INFO) << "result tensor copied = " << tensor_copied.DebugString(128) << "\n";
  }

  return Error::success();
}

// Entry point for all CPU runners. Expects the common argc/argv arguments for
// standard C++ main functions and an mlirTransformer.
// The latter is applied after parsing the input into MLIR IR and before passing
// the MLIR module to the ExecutionEngine.
int JitRunnerMain(
    int argc, char **argv,
    llvm::function_ref<LogicalResult(mlir::ModuleOp)> mlirTransformer) {
  llvm::InitLLVM y(argc, argv);

  initializeLLVM();
  mlir::initializeLLVMPasses();

  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR CPU execution driver\n");

  llvm::Optional<unsigned> optLevel = getCommandLineOptLevel();
  llvm::SmallVector<std::reference_wrapper<llvm::cl::opt<bool>>, 4> optFlags{
      optO0, optO1, optO2, optO3};
  unsigned optCLIPosition = 0;
  // Determine if there is an optimization flag present, and its CLI position
  // (optCLIPosition).
  for (unsigned j = 0; j < 4; ++j) {
    auto &flag = optFlags[j].get();
    if (flag) {
      optCLIPosition = flag.getPosition();
      break;
    }
  }
  // Generate vector of pass information, plus the index at which we should
  // insert any optimization passes in that vector (optPosition).
  llvm::SmallVector<const llvm::PassInfo *, 4> passes;
  unsigned optPosition = 0;
  for (unsigned i = 0, e = llvmPasses.size(); i < e; ++i) {
    passes.push_back(llvmPasses[i]);
    if (optCLIPosition < llvmPasses.getPosition(i)) {
      optPosition = i;
      optCLIPosition = UINT_MAX; // To ensure we never insert again
    }
  }

  MLIRContext context;
  auto m = parseMLIRInput(inputFilename, &context);
  if (!m) {
    llvm::errs() << "could not parse the input IR\n";
    return 1;
  }

  if (mlirTransformer)
    if (failed(mlirTransformer(m.get())))
      return EXIT_FAILURE;

  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Failed to create a JITTargetMachineBuilder for the host\n";
    return EXIT_FAILURE;
  }
  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Failed to create a TargetMachine for the host\n";
    return EXIT_FAILURE;
  }

  auto transformer = mlir::makeLLVMPassesTransformer(
      passes, optLevel, /*targetMachine=*/tmOrError->get(), optPosition);

  // Get the function used to compile and execute the module.
  using CompileAndExecuteFnT = Error (*)(
      ModuleOp, StringRef, std::function<llvm::Error(llvm::Module *)>);
  auto compileAndExecuteFn =
      llvm::StringSwitch<CompileAndExecuteFnT>(mainFuncType.getValue())
          .Case("f32", compileAndExecuteSingleFloatReturnFunction)
          .Case("void", compileAndExecuteVoidFunction)
          .Default(nullptr);

  Error error =
      compileAndExecuteFn
          ? compileAndExecuteFn(m.get(), mainFuncName.getValue(), transformer)
          : make_string_error("unsupported function type");

  int exitCode = EXIT_SUCCESS;
  llvm::handleAllErrors(std::move(error),
                        [&exitCode](const llvm::ErrorInfoBase &info) {
                          llvm::errs() << "Error: ";
                          info.log(llvm::errs());
                          llvm::errs() << '\n';
                          exitCode = EXIT_FAILURE;
                        });

  return exitCode;
}

// Jit run jit function with external args
//
int JitRunnerMainWithArgs(
    int argc, char **argv,
    llvm::function_ref<LogicalResult(mlir::ModuleOp)> mlirTransformer) {
  llvm::InitLLVM y(argc, argv);

  initializeLLVM();
  mlir::initializeLLVMPasses();

  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR CPU execution driver\n");

  llvm::Optional<unsigned> optLevel = getCommandLineOptLevel();
  llvm::SmallVector<std::reference_wrapper<llvm::cl::opt<bool>>, 4> optFlags{
      optO0, optO1, optO2, optO3};
  unsigned optCLIPosition = 0;
  // Determine if there is an optimization flag present, and its CLI position
  // (optCLIPosition).
  for (unsigned j = 0; j < 4; ++j) {
    auto &flag = optFlags[j].get();
    if (flag) {
      optCLIPosition = flag.getPosition();
      break;
    }
  }
  // Generate vector of pass information, plus the index at which we should
  // insert any optimization passes in that vector (optPosition).
  llvm::SmallVector<const llvm::PassInfo *, 4> passes;
  unsigned optPosition = 0;
  for (unsigned i = 0, e = llvmPasses.size(); i < e; ++i) {
    passes.push_back(llvmPasses[i]);
    if (optCLIPosition < llvmPasses.getPosition(i)) {
      optPosition = i;
      optCLIPosition = UINT_MAX; // To ensure we never insert again
    }
  }

  MLIRContext context;
  auto m = parseMLIRInput(inputFilename, &context);
  if (!m) {
    llvm::errs() << "could not parse the input IR\n";
    return 1;
  }

  if (mlirTransformer)
    if (failed(mlirTransformer(m.get())))
      return EXIT_FAILURE;

  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Failed to create a JITTargetMachineBuilder for the host\n";
    return EXIT_FAILURE;
  }
  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Failed to create a TargetMachine for the host\n";
    return EXIT_FAILURE;
  }

  auto transformer = mlir::makeLLVMPassesTransformer(
      passes, optLevel, /*targetMachine=*/tmOrError->get(), optPosition);

  // Get the function used to compile and execute the module.
  using CompileAndExecuteFnT = Error (*)(
      ModuleOp, StringRef, std::function<llvm::Error(llvm::Module *)>);
  auto compileAndExecuteFn =
      llvm::StringSwitch<CompileAndExecuteFnT>(mainFuncType.getValue())
          .Case("f32", compileAndExecuteSingleFloatReturnFunction)
          .Case("void", compileAndExecuteVoidFunction)
          .Case("user_define", compileAndExecuteFunctionWithArgs)
          .Default(nullptr);

  Error error =
      compileAndExecuteFn
          ? compileAndExecuteFn(m.get(), mainFuncName.getValue(), transformer)
          : make_string_error("unsupported function type");

  int exitCode = EXIT_SUCCESS;
  llvm::handleAllErrors(std::move(error),
                        [&exitCode](const llvm::ErrorInfoBase &info) {
                          llvm::errs() << "Error: ";
                          info.log(llvm::errs());
                          llvm::errs() << '\n';
                          exitCode = EXIT_FAILURE;
                        });

  return exitCode;
}

// Test run jit function with input args
// RUN:
// bazel-bin/tensorflow/compiler/mlir/tensorflow/tf-mlir-runtime
//   -e main_1d_i32 | main_1d_i64 | main_2d_i32 | main_2d_i64 | main_2d_f32 | main_2d_f64
//   -entry-point-result=user_define
//   -shared-libs=/path/libtf_mlir_compiler_util.so
//
int main(int argc, char **argv) {
  return JitRunnerMainWithArgs(argc, argv, nullptr);
}

