// 1) bazel build --config=opt --config=noaws --config=nohdfs   --config=nonccl  //tensorflow/compiler/mlir:tf-opt
// 2) bazel build --config=opt --config=noaws --config=nohdfs   --config=nonccl  //tensorflow/compiler/mlir/tensorflow:tf-mlir-runtime
// 3) bazel build --config=opt --config=noaws --config=nohdfs   --config=nonccl  //tensorflow/compiler/mlir:tf_mlir_external_copy_results
// 4) bazel-bin/tensorflow/compiler/mlir/tf-opt tensorflow/compiler/mlir/tensorflow/tests/mlir_runtime_results.mlir -tf-executor-to-tf-conversion --xla-legalize-tf -hlo-legalize-to-lhlo --lhlo-legalize-to-linalg --lhlo-legalize-to-std --convert-linalg-to-affine-loops  --lower-affine  --convert-loop-to-std -convert-tf-to-llvm --convert-std-to-llvm | bazel-bin/tensorflow/compiler/mlir/tensorflow/tf-mlir-runtime -e main_ret2 -entry-point-result=user_define -shared-libs=bazel-bin/tensorflow/compiler/mlir/libtf_mlir_external_copy_results.so
// the function name can be: -e main_ret1 ; -e main_ret2 ; -e main_ret3

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 134 : i32}} {
  func @main_ret1(%arg0: tensor<?xi64>, %arg1: tensor<1xi64>) {
    tf_executor.graph {
      %src1, %src_ctl1 = tf_executor.island wraps "tf.Const"() {name = "Const", value = dense<[[8,5], [9,3]]> : tensor<2x2xi32>} : () -> tensor<?x?xi32>
      %cr, %dest_ct2 = tf_executor.island wraps "tf.CopyResult"(%arg1, %src1) {} : (tensor<1xi64>, tensor<?x?xi32>) -> (tensor<1xi64>)

      %src2, %src_ctl2 = tf_executor.island wraps "tf.Const"() {name = "Const", value = dense<[[8,5,5], [9,3,3], [1,1,1]]> : tensor<3x3xi32>} : () -> tensor<3x3xi32>
      %cr4, %dest_ct4 = tf_executor.island wraps "tf.CopyResult"(%arg1, %src2) {} : (tensor<1xi64>, tensor<3x3xi32>) -> (tensor<1xi64>)
      tf_executor.fetch
    }
    return
  }

  func @main_ret2(%arg1: tensor<1xi64>) {
    tf_executor.graph {
      %op1, %ctl1 = tf_executor.island wraps "tf.Const"() {name = "Const", value = dense<[[8,5], [9,3]]> : tensor<2x2xi64>} : () -> tensor<?x?xi64>
      %op2, %ctl2 = tf_executor.island wraps "tf.Const"() {name = "Const", value = dense<[[8,-3], [6,0]]> : tensor<2x2xi64>} : () -> tensor<?x?xi64>
      %sub, %ctl3 = tf_executor.island wraps "tf.Sub"(%op1, %op2) {T = i64, device = "CPU:0"} : (tensor<?x?xi64>, tensor<?x?xi64>) -> (tensor<?x?xi64>)
      %cr, %dest = tf_executor.island wraps "tf.CopyResult"(%arg1, %sub) {} : (tensor<1xi64>, tensor<?x?xi64>) -> (tensor<1xi64>)
      tf_executor.fetch
    }
    return
  }

  func @main_ret3(%arg0: tensor<?x?xf64>, %arg1: tensor<?x?xf64>, %ret0: tensor<1xi64>) {
    tf_executor.graph {
      %cr, %dest_ctl = tf_executor.island wraps "tf.CopyResult"(%ret0, %arg0) {} : (tensor<1xi64>, tensor<?x?xf64>) -> (tensor<1xi64>)
      tf_executor.fetch
    }
    return
  }

}

