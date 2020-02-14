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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::InferenceContext;

REGISTER_OP("_MlirRun")
    .Input("args: Targs")
    .Attr("Targs: list(type) >= 0")
    .Output("results: Tresults")
    .Attr("Tresults: list(type) >= 0")
    .Attr("compiled_func_name: string")
    .SetIsStateful()
    .Doc(R"(MLIR Run Op. For use by the MLIR JIT only.

Executes a TensorFlow function previously compiled into a Mlir executable.
)");

}  // namespace tensorflow

