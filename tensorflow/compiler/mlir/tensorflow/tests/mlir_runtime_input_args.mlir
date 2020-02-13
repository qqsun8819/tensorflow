// 1) bazel build --config=opt --config=noaws --config=nohdfs   --config=nonccl  //tensorflow/compiler/mlir:tf-opt
// 2) bazel build --config=opt --config=noaws --config=nohdfs   --config=nonccl  //tensorflow/compiler/mlir/tensorflow:tf-mlir-runtime
// 3) bazel build --config=opt --config=noaws --config=nohdfs   --config=nonccl  //tensorflow/compiler/mlir:tf_mlir_compiler_util
// 4) bazel-bin/tensorflow/compiler/mlir/tf-opt -tf-executor-to-tf-conversion -convert-tf-to-affine --convert-tf-to-llvm mlir_runtime_input_args.mlir | bazel-bin/tensorflow/compiler/mlir/tensorflow/tf-mlir-runtime -e main_1d_i64 -entry-point-result=user_define -shared-libs=/home/jiankeng.pt/workspace/tensorflow-bak/bazel-bin/tensorflow/compiler/mlir/libtf_mlir_compiler_util.so
// the function name can be: -e main_1d_i64, -e main_2d_f32, -e main_2d_i32, -e main_2d_f64

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 134 : i32}} {
  func @main_1d_i64(%arg0: memref<?xi64>) {
    tf_executor.graph {
      %outputs_2, %control_3 = tf_executor.island wraps "tf.CallExternalFunc3"(%arg0, %arg0) {device = "", name = "", value_dims = dense<1> : tensor<i32>} : (memref<?xi64>, memref<?xi64>) -> tensor<1xi32>
      tf_executor.fetch
    }
    return
  }

  func @main_2d_i64(%arg0: memref<?x?xi64>) {
    tf_executor.graph {
      %outputs_2, %control_3 = tf_executor.island wraps "tf.CallExternalFunc3"(%arg0, %arg0) {device = "", name = "", value_dims = dense<2> : tensor<i32>} : (memref<?x?xi64>, memref<?x?xi64>) -> tensor<1xi32>
      tf_executor.fetch
    }
    return
  }

  func @main_1d_i32(%arg0: memref<?xi32>) {
    tf_executor.graph {
      %outputs_2, %control_3 = tf_executor.island wraps "tf.CallExternalFunc3"(%arg0, %arg0) {device = "", name = "", value_dims = dense<1> : tensor<i32>} : (memref<?xi32>, memref<?xi32>) -> tensor<1xi32>
      tf_executor.fetch
    }
    return
  }

  func @main_2d_i32(%arg0: memref<?x?xi32>) {
    tf_executor.graph {
      %outputs_2, %control_3 = tf_executor.island wraps "tf.CallExternalFunc3"(%arg0, %arg0) {device = "", name = "", value_dims = dense<2> : tensor<i32>} : (memref<?x?xi32>, memref<?x?xi32>) -> tensor<1xi32>
      tf_executor.fetch
    }
    return
  }

  func @main_2d_f64(%arg0: memref<?x?xf64>) {
    tf_executor.graph {
      %outputs_2, %control_3 = tf_executor.island wraps "tf.CallExternalFunc3"(%arg0, %arg0) {device = "", name = "", value_dims = dense<2> : tensor<i32>} : (memref<?x?xf64>, memref<?x?xf64>) -> tensor<1xi32>
      tf_executor.fetch
    }
    return
  }

  func @main_2d_f32(%arg0: memref<?x?xf32>) {
    tf_executor.graph {
      %outputs_2, %control_3 = tf_executor.island wraps "tf.CallExternalFunc3"(%arg0, %arg0) {device = "", name = "", value_dims = dense<2> : tensor<i32>} : (memref<?x?xf32>, memref<?x?xf32>) -> tensor<1xi32>
      tf_executor.fetch
    }
    return
  }

}
