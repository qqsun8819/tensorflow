// bazel-bin/tensorflow/compiler/mlir/tf-opt tensorflow/compiler/mlir/tensorflow/tests/unique.mlir -tf-executor-to-tf-conversion   --xla-legalize-tf -hlo-legalize-to-lhlo --lhlo-legalize-to-linalg --lhlo-legalize-to-std --convert-linalg-to-affine-loops  --lower-affine --convert-loop-to-std - --convert-std-to-llvm|bazel-bin/tensorflow/compiler/mlir/tensorflow/tf-mlir-runtime -e main_unqiue_1d64 -entry-point-result=user_define -shared-libs=bazel-bin/tensorflow/compiler/mlir/libtf_mlir_compiler_util.so

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 134 : i32}} {
  func @main_unqiue_1d64(%arg0: tensor<?xi64>) {
    tf_executor.graph {
      %u1, %u2, %control_1 = tf_executor.island wraps "tf.Unique"(%arg0)  : (tensor<?xi64>) -> (tensor<?xi64>, tensor<?xi64>)
      %control_2 = tf_executor.island wraps "tf.DebugPrint"(%arg0)  : (tensor<?xi64>) -> ()
      %control_4 = tf_executor.island wraps "tf.DebugPrint"(%u1)  : (tensor<?xi64>) -> ()
      %control_5 = tf_executor.island wraps "tf.DebugPrint"(%u2)  : (tensor<?xi64>) -> ()
      
      tf_executor.fetch
    }
    return
  }
}

