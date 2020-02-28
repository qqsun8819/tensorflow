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


#include "tensorflow/compiler/jit/mlir_compiler.h"

#include <vector>
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "tensorflow/compiler/jit/shape_inference.h"
#include "tensorflow/compiler/mlir/tensorflow/runtime/dynamic_memref.h"
#include "tensorflow/compiler/mlir/tf_mlir_compiler.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"

#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace  {
NameAttrList FunctionAttr(OpKernelConstruction* ctx) {
  const NameAttrList* func;
  if(!ctx->GetAttr("function", &func).ok())
    return NameAttrList();
  return *func;
}


std::vector<int> ConstantsVector(OpKernelConstruction* ctx) {
  DataTypeVector constant_types;
  if (!ctx->GetAttr("Tconstants", &constant_types).ok())
    return std::vector<int>();
  std::vector<int> constants(constant_types.size());
  std::iota(constants.begin(), constants.end(), 0);
  return constants;
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



}
MlirCompiler::MlirCompiler(OpKernelConstruction* ctx)
    : device_(new XlaCompilationDevice(SessionOptions(), ctx->device_type())),
      device_mgr_(absl::WrapUnique(device_)),
      function_(FunctionAttr(ctx)),
      constants_(ConstantsVector(ctx)) { 
  flib_def_ = ctx->function_library()->GetFunctionLibraryDefinition();
  pflr_.reset(new ProcessFunctionLibraryRuntime(
      &device_mgr_, Env::Default(), /*config=*/nullptr,
      ctx->function_library()->graph_def_version(), flib_def_, OptimizerOptions()));
  flib_runtime_ = pflr_->GetFLR(device_->name());
}

static Status GetFunctionBody(const NameAttrList& function,
                              FunctionLibraryRuntime* flib_runtime,
                              const FunctionBody** fbody) {
  FunctionLibraryRuntime::Handle handle;
  TF_RETURN_IF_ERROR(flib_runtime->Instantiate(
      function.name(), AttrSlice(&function.attr()), &handle));

  *fbody = flib_runtime->GetFunctionBody(handle);
  if (!(*fbody))
    return errors::Internal("fbody get error");
  return Status::OK();
}

Status MlirCompiler::FindFunctionBody(const NameAttrList& function,
                                     const FunctionBody** fbody) {
  // The function may be in either the local_flib_runtime_ or flib_runtime_.
  // Look up the function in local first and if it is not found then look up the
  // function in flib_runtime_.
  auto status = GetFunctionBody(function, flib_runtime_, fbody);
  if (!status.ok()) {
    return status;
  }
  LOG(INFO) << "Function " << function.name() << " in flib_runtime_";
  return Status::OK();
}

std::unique_ptr<Graph> MlirCompiler::GetGraph(const FunctionBody* fbody) {
  std::unique_ptr<Graph> graph(new Graph(flib_def_));
  CopyGraph(*fbody->graph, graph.get());

  // Performs a first function inlining pass before shape inference, since
  // otherwise shape inference can't see inside functions and a comprehensive
  // shape_map, including function ops, is needed to constant-propagate Shape
  // Ops below.
  OptimizerOptions opts;
  opts.set_opt_level(OptimizerOptions::L0);
  opts.set_do_common_subexpression_elimination(false);
  opts.set_do_function_inlining(true);
  opts.set_do_constant_folding(true);
  GraphOptimizer optimizer(opts);
  // Do not constant fold nodes that output DT_VARIANT type tensors.
  // XLA does not support Const nodes of Variant type since it needs
  // to know the original ops to be able to compile them to the relevant
  // XLA form.
  // TODO(srbs): This filter is a little conservative. E.g. a subgraph of
  // the form:
  //                          Const
  //                            |
  // EmptyTensorList -> TensorListPushBack -> TensorListPopBack -> Op
  //                                                  |
  //                                        (Discard popped list)
  //
  // Would have been reduced to "Const -> Op" without this filter.
  // However since we are only allowed to specify the filter at the "Node"
  // level there is no good way to allow the above behavior. So we
  // disallow any sort of constant folding on Variant nodes for now.
  //
  // Also do not consider constant folding Shape ops. When there is a dynamic
  // dimension in a tensor, TF2XLA currently represent them as the static
  // upperbound shape, which can be constant folded and then lose the info
  // that this Shape is dynamic.
  auto cf_consider_fn = [](const Node* n) {
    for (const auto& output_arg : n->op_def().output_arg()) {
      if (output_arg.type() == DT_VARIANT) {
        return false;
      }
    }
    const auto& ts = n->type_string();
    // XLA has special logic to handle dynamic shapes, don't constant fold
    // them.
    if (ts == "Shape" || ts == "ShapeN" || ts == "Size") {
      return false;
    }
    return true;
  };
  GraphOptimizer::Options graph_optimizer_options;
  graph_optimizer_options.cf_consider_fn = cf_consider_fn;
  graph_optimizer_options.inline_multi_device_functions = true;
  graph_optimizer_options.inline_impl_selection_group_functions = true;
  optimizer.Optimize(flib_runtime_, flib_runtime_->env(),
                     /*device=*/nullptr, &graph, graph_optimizer_options);

  // Run shape inference on the graph and optimize the graph again.
  GraphShapeInfo shape_info;
  InferShapes(graph.get(), /*arg_shapes=*/{},
              flib_runtime_->GetFunctionLibraryDefinition(), &shape_info)
      .IgnoreError();
  auto node_name_index = graph->BuildNodeNameIndex();
  std::unordered_map<string, std::vector<PartialTensorShape>> shape_map;
  for (const auto& node_shape_info : shape_info) {
    const string& node_name = node_shape_info.first;
    const std::vector<InferredShape>& output_shapes = node_shape_info.second;
    const auto& node_iter = node_name_index.find(node_name);
    if (node_iter != node_name_index.end()) {
      auto& partial_shapes = shape_map[node_name];
      for (const auto& inferred_shape : output_shapes) {
        partial_shapes.push_back(inferred_shape.shape);
      }
    }
  }
  graph_optimizer_options.shape_map = &shape_map;
  optimizer.Optimize(flib_runtime_, flib_runtime_->env(),
                     /*device=*/nullptr, &graph, graph_optimizer_options);

  return graph;
}

Status MlirCompiler::ProcessConstArgNodes(std::unique_ptr<Graph>* graph) {
  Graph* g = graph->get();
  std::vector<Node*> arg_nodes;
  absl::c_copy_if(g->op_nodes(), std::back_inserter(arg_nodes),
                  [](const Node* n) {
                    if (n->def().op() == "_Arg") {
                      return true;
                    } else {
                      return false;
                    }
                  });
  LOG(INFO) << "arg size:" << arg_nodes.size();

  int num_constant_inputs;
  Status status;
  status = GetNodeAttr(AttrSlice(&function_.attr()), "_XlaNumConstantArgs", &num_constant_inputs);
  if (!status.ok())
    return status;

  for (Node* n : arg_nodes) {
    DataType node_type;
    if (TryGetNodeAttr(n->attrs(), "_dtype", &node_type)) {
      LOG(INFO) << "replace node for arg";
      Scope scope = NewInternalScope(g, &status, /*refiner=*/nullptr)
        .NewSubScope(n->name())
        .WithDevice(n->requested_device())
        .WithAssignedDevice(n->requested_device());
      if (!status.ok()) {
        return status;
      }

      ::tensorflow::Node* ret = nullptr;
      const TensorProto* node_value = nullptr;
      if (!GetNodeAttr(n->attrs(), "_value", &node_value).ok())
        LOG(FATAL) << "Insert Const value node to graph failed.";

      const auto unique_name = scope.GetUniqueNameForOp("Const");
      auto builder = ::tensorflow::NodeBuilder(unique_name, "Const")
        .Attr("dtype", node_type)
        .Attr("value", *node_value);

      scope.UpdateBuilder(&builder);
      scope.UpdateStatus(builder.Finalize(g, &ret));

      if (!scope.ok()) {
        return scope.status();
      }

      // copy output edges
      ReplaceOutEdges(g, n, ret);
      g->RemoveNode(n);
    } else {
      // process non-constant _Arg
      int index;
      status = GetNodeAttr(n->attrs(), "index", &index);
      if (!status.ok())
        return status;
        
      n->ClearAttr("index");
      n->AddAttr("index", index -  num_constant_inputs); 
       
    }

  }
  return Status::OK();
}   

Status MlirCompiler::CompileGraph(OpKernelContext* ctx, const std::string& func_name) {

  if (MlirExecutableClosureStore::Global()->Exists(func_name)) 
    return Status::OK();

  const FunctionBody* fbody;
  TF_RETURN_IF_ERROR(FindFunctionBody(function_, &fbody));

  std::map<int, Tensor> constant_args;
  for (int i : constants_) {
    constant_args.insert({i, ctx->input(i)});
  }
  for (int i = 0; i < ctx->num_inputs(); i++) {
    DataType dtype;
    TF_RETURN_IF_ERROR(GetNodeAttr(fbody->arg_nodes[i]->def(), "T", &dtype));
    if (dtype == DT_RESOURCE || dtype == DT_VARIANT) {
      continue;
    }

    if (constant_args.count(i) > 0) {
      const Tensor& input = constant_args.at(i);
      fbody->arg_nodes[i]->ClearAttr("_output_shapes");
      fbody->arg_nodes[i]->AddAttr("_output_shapes",
          std::vector<TensorShape>{input.shape()});
      fbody->arg_nodes[i]->AddAttr("_value", input);
      fbody->arg_nodes[i]->AddAttr("_dtype", input.dtype());
      fbody->arg_nodes[i]->AddAttr("_shape", input.shape());
    } else {
      const Tensor& input = ctx->input(i);
      TensorShapeProto arg_shape ; 
      for (int i = 0;i < input.dims(); i++) {
        arg_shape.add_dim()->set_size(-1);
      }
      fbody->arg_nodes[i]->ClearAttr("_output_shapes");
      fbody->arg_nodes[i]->AddAttr("_output_shapes",
          gtl::ArraySlice<TensorShapeProto>{arg_shape});
    }
  }
  std::unique_ptr<Graph> graph = GetGraph(fbody);
  TF_RETURN_IF_ERROR(ProcessConstArgNodes(&graph));
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "XlaCompiler::CompileFunction: "
            << DumpGraphToFile(
                   absl::StrCat("mlir_compile_function_", function_.name()), *graph);
  }

  GraphDef graph_def;
  graph->ToGraphDef(&graph_def);

  MlirExecutableClosureStore::Global()->Produce(func_name, graph_def, "");
  return Status::OK();
}



} // namespace tensorflow
