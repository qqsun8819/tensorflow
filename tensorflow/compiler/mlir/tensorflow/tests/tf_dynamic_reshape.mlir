// 1) bazel build --config=opt --config=noaws --config=nohdfs   --config=nonccl  //tensorflow/compiler/mlir:tf-opt
// 2) bazel build --config=opt --config=noaws --config=nohdfs   --config=nonccl  //tensorflow/compiler/mlir:tf_mlir_compiler_util
// 3) bazel-bin/tensorflow/compiler/mlir/tf-opt -tf-executor-to-tf-conversion -convert-tf-to-affine -convert-tf-to-llvm  tf_dynamic_reshape.mlir

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 134 : i32}} {
  func @main() {
    tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.Const"() {device = "", dtype = "tfdtype$DT_INT64", name = "Const", value = dense<[1,2,3,4,5,6,7,8,9]> : tensor<9xi64>} : () -> tensor<9xi64>
      %outputs2, %control2 = tf_executor.island wraps "tf.Const"() {device = "", dtype = "tfdtype$DT_INT64", name = "Const2", value = dense<[3, 3]> : tensor<2xi64>} : () -> tensor<2xi64>

      %u1, %u2 = tf_executor.island wraps "tf.Reshape"(%outputs, %outputs2) {device = "", name = "MIND/xxx/Reshape"} : (tensor<9xi64>, tensor<2xi64>) -> tensor<3x?xi64>
      %u11, %u22 = tf_executor.island wraps "tf.Reshape"(%outputs, %outputs2) {device = "", name = "MIND/xxx/Reshape"} : (tensor<9xi64>, tensor<2xi64>) -> tensor<?x?xi64>
      %u111, %u222 = tf_executor.island wraps "tf.Reshape"(%outputs, %outputs2) {device = "", name = "MIND/xxx/Reshape"} : (tensor<9xi64>, tensor<2xi64>) -> tensor<3x3xi64>
      %3:2 = tf_executor.island wraps "tf.RawDebugPrint2"(%u1, %u1) {device = "", value_dims = dense<2> : tensor<i32>, name = ""} : (tensor<3x?xi64>, tensor<3x?xi64>) -> (tensor<1xi32>)
      %4:2 = tf_executor.island wraps "tf.RawDebugPrint2"(%u11, %u11) {device = "", value_dims = dense<2> : tensor<i32>, name = ""} : (tensor<?x?xi64>, tensor<?x?xi64>) -> (tensor<1xi32>)
      %5:2 = tf_executor.island wraps "tf.RawDebugPrint2"(%u111, %u111) {device = "", value_dims = dense<2> : tensor<i32>, name = ""} : (tensor<3x3xi64>, tensor<3x3xi64>) -> (tensor<1xi32>)

      tf_executor.fetch
    }
    return
  }
}

