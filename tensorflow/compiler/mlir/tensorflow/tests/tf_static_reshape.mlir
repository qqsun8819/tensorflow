// RUN: bazel-bin/tensorflow/compiler/mlir/tf-opt -tf-executor-to-tf-conversion -convert-tf-to-affine -convert-tf-to-llvm  tf_static_reshape.mlir

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 134 : i32}} {
  func @main(%arg0: i64) {
    tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.Const"() {device = "", dtype = "tfdtype$DT_INT64", name = "Const", value = dense<[1,2,3,4,5,6,7,8]> : tensor<8xi64>} : () -> tensor<8xi64>
      %outputs2, %control2 = tf_executor.island wraps "tf.Const"() {device = "", dtype = "tfdtype$DT_INT64", name = "Const", value = dense<[4, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
      %output3, %control3 = tf_executor.island wraps "tf.Reshape"(%outputs, %outputs2) {device = "", name = "MIND/xxx/Reshape"} : (tensor<8xi64>, tensor<2xi64>) -> (tensor<4x2xi64>)
      %output4, %control4 = tf_executor.island wraps "tf.RawDebugPrint"(%output3, %output3) {T = "tfdtype$DT_INT64", device = "", name = "CallExternalFunc_1"} : (tensor<4x2xi64>, tensor<4x2xi64>) -> (tensor<4x2xi64>)
      tf_executor.fetch
    }
    return
  }
}

