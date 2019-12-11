/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <stdio.h>
#include <memory>
#include <string>

#include "absl/types/span.h"
#include "tensorflow/stream_executor/tf_allocator_adapter.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/jit/xla_tensor.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/service.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace tools {


template <typename T>
void FillFn(tensorflow::Tensor* tensor, std::function<T(int)> fn) {
  auto flat = tensor->flat<T>();
  for (int i = 0; i < flat.size(); ++i) flat(i) = fn(i);
}

StatusOr<std::unique_ptr<HloModule>> BuildHloModule(XlaBuilder* b) {
  TF_ASSIGN_OR_RETURN(XlaComputation computation,
      b->Build(/*remove_dynamic_dimensions=*/false));
  const HloModuleProto& proto = computation.proto();
  TF_ASSIGN_OR_RETURN(const auto& config,
      HloModule::CreateModuleConfigFromProto(
        proto, GetDebugOptionsFromFlags()));
  return HloModule::CreateFromProto(proto, config);
}



void CompileAndRunXlaComputation(const XlaComputation& computation) {

  LocalClient* client = ClientLibrary::LocalClientOrDie();
  //LocalService* local_service =ClientLibrary::GetXlaService(client->platform());

  std::unique_ptr<ProgramShape> program_shape =
    client->GetComputationShape(computation).ConsumeValueOrDie();

  std::vector<const Shape*> layouts;
  layouts.reserve(program_shape->parameters_size());
  for (int i = 0; i < program_shape->parameters_size(); ++i) {
    layouts.push_back(&program_shape->parameters(i));
    LOG(INFO) << "arg["<< i<<"], shape:" << xla::ShapeUtil::HumanStringWithLayout(program_shape->parameters(i)) ;
  }

  ExecutableBuildOptions build_options;
  build_options.set_device_ordinal(0);
  build_options.set_result_layout(program_shape->result());
  LOG(INFO) << "result shape:" << xla::ShapeUtil::HumanStringWithLayout(program_shape->result());

  StatusOr<std::unique_ptr<LocalExecutable>> local_executable =
    client->Compile(computation, layouts, build_options);

  const HloModule& module = local_executable.ValueOrDie()->executable()->module();

  fprintf(stdout, "HLO compiled for %s backend:\n%s\n",
      client->platform()->Name().c_str(),
      module.ToString(HloPrintOptions::ShortParsable()).c_str());
  std::vector<std::unique_ptr<xla::ShapedBuffer>> arg_buffers_;
  std::vector<xla::ShapedBuffer*> arg_ptrs_;
  arg_buffers_.reserve(program_shape->parameters_size() + 1);
  arg_buffers_.resize(program_shape->parameters_size());
  arg_ptrs_ = std::vector<ShapedBuffer*>(arg_buffers_.size());

  for (int i = 0; i < program_shape->parameters_size(); ++i) {
    tensorflow::TensorShape a_shape;
    Shape xla_shape = program_shape->parameters(i); 
    for (int ai = 0; ai < xla_shape.dimensions_size(); ++ai) {
      a_shape.AddDim(xla_shape.dimensions(ai));
    }

    tensorflow::Tensor a(tensorflow::DT_INT64, a_shape);
    FillFn<int64>(&a, [](int i)->int64 { return 2; });
    LOG(INFO) << a.DebugString();

    se::DeviceMemoryBase dmem = tensorflow::XlaTensor::DeviceMemoryFromTensor(a);
    arg_buffers_[i] = absl::make_unique<ShapedBuffer>(
        /*on_host_shape=*/xla_shape, 
        /*on_device_shape=*/xla_shape,
        client->platform(), client->default_device_ordinal());
    arg_buffers_[i]->set_buffer(dmem, /*index=*/{});
    arg_ptrs_[i] = arg_buffers_[i].get();
  }


  absl::optional<se::TfAllocatorAdapter> tf_allocator_adapter;
  tf_allocator_adapter.emplace(tensorflow::cpu_allocator(), client->platform());
  se::DeviceMemoryAllocator* allocator =
    &tf_allocator_adapter.value();
  xla::ExecutableRunOptions run_options;
  run_options.set_stream(nullptr);
  run_options.set_allocator(allocator);
  run_options.set_intra_op_thread_pool(nullptr);
  run_options.set_rng_seed(tensorflow::GetXLARandomSeed());


  local_executable.ValueOrDie()->Run(arg_ptrs_, run_options);
}

void BuildDynamicShape() {
  XlaBuilder b("tool_xlabuilder");
  Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {5}, {true}),
       ShapeUtil::MakeShape(F32, {5}, {true}), 
       ShapeUtil::MakeShape(U32, {})}
       );
 
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  TF_CHECK_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{2},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{0},
                                   /*target_dim_num=*/0));
  TF_CHECK_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{2},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{1},
                                   /*target_dim_num=*/0));
  auto gte0 = GetTupleElement(p0, 0);
  auto gte1 = GetTupleElement(p0, 1);
  Add(gte0, gte1);

  auto computation_status = b.Build(/*remove_dynamic_dimensions=*/false);
  if (computation_status.status().ok()) {
    XlaComputation computation = computation_status.ConsumeValueOrDie(); 
    CompileAndRunXlaComputation(computation);
  }

}
}  // namespace tools
}  // namespace xla

int main(int argc, char** argv) {
  bool compile = true;
  std::vector<tensorflow::Flag> flag_list = {
      {"compile", &compile,
       "If true, compile the computation using the default client before "
       "dumping the HLO. Otherwise dump the raw (uncompiled) HLO."},
  };
  const xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  bool parsed_flags_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  QCHECK(parsed_flags_ok) << "\n" << usage;

  tensorflow::port::InitMain(usage.c_str(), &argc, &argv);
  QCHECK(argc > 1) << "\nERROR: must specify at least one module\n" << usage;

  absl::Span<char* const> args(argv, argc);
  args.remove_prefix(1);  // Pop off the binary name, argv[0]
  xla::tools::BuildDynamicShape();

  for (char* arg : args) {
    xla::HloSnapshot snapshot;
    TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(), arg,
                                            &snapshot));

    if(snapshot.has_hlo() && snapshot.hlo().has_hlo_module()) {
      xla::XlaComputation computation(snapshot.hlo().hlo_module());
      xla::tools::CompileAndRunXlaComputation(computation);
    } else {
      LOG(ERROR) << "snapshot not correct"; 
    }
  }
  return 0;
}
