// 1) bazel build --config=opt --config=noaws --config=nohdfs   --config=nonccl  //tensorflow/compiler/mlir:tf-opt
// 2) bazel build --config=opt --config=noaws --config=nohdfs   --config=nonccl  //tensorflow/compiler/mlir/tensorflow:tf-mlir-runtime
// 3) bazel build --config=opt --config=noaws --config=nohdfs   --config=nonccl  //tensorflow/compiler/mlir:tf_mlir_compiler_util
// 4) bazel-bin/tensorflow/compiler/mlir/tf-opt -tf-executor-to-tf-conversion -tf-executor-to-tf-conversion --xla-legalize-tf  -hlo-legalize-to-lhlo --lhlo-legalize-to-std  --convert-std-to-llvm  tensorflow/compiler/mlir/tensorflow/tests/mlir_runtime_input_args.mli | bazel-bin/tensorflow/compiler/mlir/tensorflow/tf-mlir-runtime -e main_1d_i64 -entry-point-result=user_define -shared-libs=bazel-bin/tensorflow/compiler/mlir/libtf_mlir_compiler_util.so
// the function name can be: -e main_1d_i64, -e main_2d_f32, -e main_2d_i32, -e main_2d_f64

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 134 : i32}} {
  func @main_1d_i64(%arg0: tensor<?xi64>) {
    tf_executor.graph {
      %control_3 = tf_executor.island wraps "tf.DebugPrint"(%arg0)  : (tensor<?xi64>) -> ()
      tf_executor.fetch
    }
    return
  }

  func @main_2d_i64(%arg0: tensor<?x?xi64>) {
    tf_executor.graph {
      %control_3 = tf_executor.island wraps "tf.DebugPrint"(%arg0)  : (tensor<?x?xi64>) -> ()
      tf_executor.fetch
    }
    return
  }

  func @main_1d_i32(%arg0: tensor<?xi32>) {
    tf_executor.graph {
      %control_3 = tf_executor.island wraps "tf.DebugPrint"(%arg0)  : (tensor<?xi32>) -> ()
      tf_executor.fetch
    }
    return
  }

  func @main_2d_i32(%arg0: tensor<?x?xi32>) {
    tf_executor.graph {
      %control_3 = tf_executor.island wraps "tf.DebugPrint"(%arg0)  : (tensor<?x?xi32>) -> ()
      tf_executor.fetch
    }
    return
  }

  func @main_2d_f64(%arg0: tensor<?x?xf64>) {
    tf_executor.graph {
      %control_3 = tf_executor.island wraps "tf.DebugPrint"(%arg0)  : (tensor<?x?xf64>) -> ()
      tf_executor.fetch
    }
    return
  }

  func @main_2d_f32(%arg0: tensor<?x?xf32>) {
    tf_executor.graph {
      %control_3 = tf_executor.island wraps "tf.DebugPrint"(%arg0)  : (tensor<?x?xf32>) -> ()
      tf_executor.fetch
    }
    return
  }

}
