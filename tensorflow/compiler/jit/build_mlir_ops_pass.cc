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
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"

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

NodeBuilder::NodeOut IncomingEdgeAsOutput(const Edge* e) {
  return NodeBuilder::NodeOut(e->src(), e->src_output());
}

void ReplaceOutEdges(Graph* graph, Node* o, Node* n) {
  std::vector<const Edge*> out_edges(
      o->out_edges().begin(),
      o->out_edges().end());
  for (const Edge* edge : out_edges) {
    graph->AddEdge(n, edge->src_output(), edge->dst(), edge->dst_input());
    graph->RemoveEdge(edge);
  }
}

Status ReplaceNodeWithMlirRun(
    const GraphOptimizationPassOptions& options,
    Graph* g, Node* n) {
  // Insert MlirRun node and delete cluster_N below.

  // For cluster_N node, the related compiled func name
  // is `cluster_N_main`
  string entry_func_name = n->name() + "main";

  // TODO: FIXME
  // Don't distinguish const/non-const/resource inputs here
  std::vector<const Edge*> input_edges_vector;
  TF_RETURN_IF_ERROR(n->input_edges(&input_edges_vector));
  absl::Span<const Edge*> input_edges(input_edges_vector);
  std::vector<NodeBuilder::NodeOut> args_inputs;
  // copy input edges
  absl::c_transform(input_edges.subspan(0, n->num_inputs()),
                    std::back_inserter(args_inputs),
                    IncomingEdgeAsOutput);
  Status status;
  Scope scope = NewInternalScope(g, &status, /*refiner=*/nullptr)
                   .NewSubScope(n->name());
  if (!status.ok()) {
    LOG(FATAL) << "Create graph scope failed.";
  }

  //ops::_MlirRun mlir_run(root.WithOpName("mlir_run"),
  //                      args_inputs);
  //mlir_run.operation.node()->AddAttr("CompiledFuncName", entry_func_name);

  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("_MlirRun");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "_MlirRun")
                       .Input(args_inputs)
                       .Attr("CompiledFuncName", entry_func_name)
                       .Attr("Targs", n->input_types())
                       .Attr("Tresults", n->output_types());

  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(g, &ret));
  if (!scope.ok()) {
    LOG(FATAL) << "Insert _MlirRun node to graph failed.";
  }
  scope.UpdateStatus(scope.DoShapeInference(ret));

  // TODO: FIXME handle control edges here

  // copy output edges
  ReplaceOutEdges(g, n, ret);
  g->RemoveNode(n);

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


