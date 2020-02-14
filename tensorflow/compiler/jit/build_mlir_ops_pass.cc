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

#include "tensorflow/compiler/jit/build_mlir_ops_pass.h"

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

namespace {

const char* const kMlirCompiledKernelAttr = "_MlirCompiledKernel";

bool IsMlirCompiledKernel(const Node& node) {
  bool is_compiled = false;
  bool has_compilation_attr =
      TryGetNodeAttr(node.attrs(), kMlirCompiledKernelAttr, &is_compiled) &&
      is_compiled;
  return has_compilation_attr ? is_compiled : false;
}

Status ReplaceNodeWithMlirRun(
    const GraphOptimizationPassOptions& options,
    Graph* g, Node* n) {
  //TODO: Insert MlirRun node here

  return Status::OK();
}

}

Status BuildMlirOpsPass::Run(const GraphOptimizationPassOptions& options) {
  Graph* graph = options.graph->get();
  // Copy out the nodes we want to rewrite to avoid modifying the graph while we
  // iterate on graph->op_nodes().
  std::vector<Node*> mlir_compiled_kernels;
  absl::c_copy_if(graph->op_nodes(), std::back_inserter(mlir_compiled_kernels),
                  [](const Node* n) {
                    // Only compile nodes that are marked for compilation by the
                    // compilation-marking pass (via 'attr_name').
                    return IsMlirCompiledKernel(*n);
                  });

  // insert MlirRun nodes
  for (Node* n : mlir_compiled_kernels) {
    TF_RETURN_IF_ERROR(ReplaceNodeWithMlirRun(
        options, graph, n));
  }

  return Status::OK();
}

}  // namespace tensorflow


