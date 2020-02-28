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
#include "tensorflow/compiler/tf2xla/cc/ops/mlir_jit_ops.h"
#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/util/dump_graph.h"

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


void ReplaceOutEdges(Graph* graph, Node* o, Node* n) {
  std::vector<const Edge*> out_edges(
      o->out_edges().begin(),
      o->out_edges().end());
  for (const Edge* edge : out_edges) {
    graph->AddEdge(n, edge->src_output(), edge->dst(), edge->dst_input());
    graph->RemoveEdge(edge);
  }
}

struct MlirClusterInfo {
  std::vector<Output> constant_inputs;
  std::vector<Output> non_constant_inputs;
  std::vector<Output> resource_inputs;
  NameAttrList function;
};

Output IncomingEdgeAsOutput(const Edge* e) {
  return Output(e->src(), e->src_output());
}

Status GetMlirClusterInfo(Node* n, MlirClusterInfo* result) {
  int num_constant_inputs, num_resource_inputs;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(n->attrs(), kXlaNumConstantArgsAttr, &num_constant_inputs));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(n->attrs(), kXlaNumResourceArgsAttr, &num_resource_inputs));

  if (num_constant_inputs < 0 || num_resource_inputs < 0 ||
      num_constant_inputs + num_resource_inputs > n->num_inputs()) {
    return errors::InvalidArgument(
        "Invalid number of constant/resource arguments to XLA kernel.");
  }

  int num_non_constant_inputs =
      n->num_inputs() - num_constant_inputs - num_resource_inputs;

  std::vector<const Edge*> input_edges_vector;
  TF_RETURN_IF_ERROR(n->input_edges(&input_edges_vector));
  absl::Span<const Edge*> input_edges(input_edges_vector);

  absl::c_transform(input_edges.subspan(0, num_constant_inputs),
                    std::back_inserter(result->constant_inputs),
                    IncomingEdgeAsOutput);

  absl::c_transform(
      input_edges.subspan(num_constant_inputs, num_non_constant_inputs),
      std::back_inserter(result->non_constant_inputs), IncomingEdgeAsOutput);

  absl::c_transform(
      input_edges.subspan(num_constant_inputs + num_non_constant_inputs,
                          num_resource_inputs),
      std::back_inserter(result->resource_inputs), IncomingEdgeAsOutput);

  result->function.set_name(n->type_string());
  *result->function.mutable_attr() = n->def().attr();
  return Status::OK();
}
Status ReplaceNodeWithMlirRun(
    const GraphOptimizationPassOptions& options,
    Graph* g, Node* n) {
  // Insert MlirRun node and delete cluster_N below.

  // For cluster_N node, the related compiled func name
  // is `cluster_Nmain`
  string entry_func_name = n->name() + "main";
  MlirClusterInfo cluster_info;
  TF_RETURN_IF_ERROR(GetMlirClusterInfo(n, &cluster_info));



  Status status;
  Scope root = NewInternalScope(g, &status, /*refiner=*/nullptr)
                   .NewSubScope(n->name())
                   .WithDevice(n->requested_device())
                   .WithAssignedDevice(n->requested_device());
  if (!status.ok()) {
    return status;
  }
  
  ops::_MlirRun mlir_run(root.WithOpName("mlir_run"),
                                /*constants=*/cluster_info.constant_inputs,
                               /*args=*/cluster_info.non_constant_inputs,
                               /*resources=*/cluster_info.resource_inputs,
                               /*Tresults=*/n->output_types(),
                               /*CompileFuncName=*/entry_func_name,
                               /*function=*/cluster_info.function);
     
          
  
  if (!root.ok()) {
    return root.status();
  }
  // scope.UpdateStatus(scope.DoShapeInference(ret));

  // TODO: FIXME handle control edges here

  // copy output edges
  ReplaceOutEdges(g, n, mlir_run.operation.node());
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
  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("build_mlir_ops", *graph, options.flib_def);
  }


  return Status::OK();
}

}  // namespace tensorflow

