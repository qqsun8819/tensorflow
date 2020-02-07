
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 175 : i32}} {
  func @main(%arg0: memref<?xi64>) {
  	// %tmp1 = alloc() : memref<8xi64>
    // %arg0 = memref_cast %tmp1: memref<8xi64> to memref<?xi64>
	%arg1 = tensor_load %arg0:memref<?xi64>
    %0 = "tf.Const"() {value = dense<34> : tensor<i64>} : () -> tensor<i64>
    %1 = "tf.Const"() {value = dense<33> : tensor<i64>} : () -> tensor<i64>
    %2 = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %4 = "tf.FloorDiv"(%arg1, %0) : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
    %5 = "tf.FloorMod"(%arg1, %0) : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
    %6 = "tf.Sub"(%arg1, %2)  : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
    %7 = "tf.FloorDiv"(%6, %1)  : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
    %8 = "tf.Maximum"(%4, %7)  : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
    %9 = "tf.Cast"(%arg1)  : (tensor<?xi64>) -> tensor<?xi32>
    %11 = "tf.Less"(%8, %2)  : (tensor<?xi64>, tensor<i64>) -> tensor<?xi1>
    %12 = "tf.FloorMod"(%6, %1)  : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
    %13 = "tf.Select"(%11, %5, %12)  : (tensor<?xi1>, tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
return 
}
}
