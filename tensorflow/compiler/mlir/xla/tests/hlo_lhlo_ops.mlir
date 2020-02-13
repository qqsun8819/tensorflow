// bazel-bin/tensorflow/compiler/mlir/tf-opt tensorflow/compiler/mlir/xla/tests/hlo_lhlo_ops.mlir --xla-legalize-tf|bazel-bin/tensorflow/compiler/mlir/tf-opt -hlo-legalize-to-lhlo --lhlo-legalize-to-linalg --lhlo-legalize-to-std --convert-linalg-to-affine-loops  --lower-affine  --convert-loop-to-std --convert-std-to-llvm|bazel-bin/external/llvm-project/mlir/mlir-cpu-runner -e main -entry-point-result=void -shared-libs=bazel-bin/tensorflow/compiler/mlir/libtf_mlir_compiler_util.so

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 134 : i32}} {
  func @main() {
	// Add/Sub/Mul/Div
    %0 = "tf.Const"() {value = dense<[[34, 35, 55], [12, 13, 33]]> : tensor<2x3xi64>} : () -> tensor<2x3xi64>
    %1 = "tf.Const"() {value = dense<[[30, 31, 11], [36, 78, 77]]> : tensor<2x3xi64>} : () -> tensor<2x3xi64>
    %11 = "tf.Const"() {value = dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi64>} : () -> tensor<2x3xi64>
    %2 = "tf.Sub"(%0, %1) {T = i64, device = "CPU:0"} : (tensor<2x3xi64>, tensor<2x3xi64>) -> (tensor<?x?xi64>)
	"tf.DebugPrint"(%2):(tensor<?x?xi64>) -> ()
    %3 = "tf.Add"(%2, %2) {T = i64, device = "CPU:0"} : (tensor<?x?xi64>, tensor<?x?xi64>) -> (tensor<?x?xi64>)
	"tf.DebugPrint"(%3):(tensor<?x?xi64>) -> ()
    %4 = "tf.Mul"(%2, %3) {T = i64, device = "CPU:0"} : (tensor<?x?xi64>, tensor<?x?xi64>) -> (tensor<?x?xi64>)
	"tf.DebugPrint"(%4):(tensor<?x?xi64>) -> ()
    %5 = "tf.Div"(%4, %11) {T = i64, device = "CPU:0"} : (tensor<?x?xi64>, tensor<2x3xi64>) -> (tensor<?x?xi64>)
	"tf.DebugPrint"(%5):(tensor<?x?xi64>) -> ()
 
    %6 = "tf.Const"() {value = dense<[[34.1, 35.2, 55.3], [12.4, 13.5, 33.6]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    %7 = "tf.Const"() {value = dense<[[4.7, 3.8, 5.1], [2.2, 1.3, 3.4]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    %8 = "tf.Sub"(%6, %7) {T = f64, device = "CPU:0"} : (tensor<2x3xf64>, tensor<2x3xf64>) -> (tensor<?x?xf64>)
	"tf.DebugPrint"(%8):(tensor<?x?xf64>) -> ()
    %9 = "tf.Neg"(%8) : (tensor<?x?xf64>) -> ( tensor<?x?xf64>)
	"tf.DebugPrint"(%9):(tensor<?x?xf64>) -> ()
    %91 = "tf.Neg"(%0) : (tensor<2x3xi64>) -> ( tensor<?x?xi64>)
	"tf.DebugPrint"(%91):(tensor<?x?xi64>) -> ()
 
    %62 = "tf.Const"() {value = dense<[[34.1, 35.2, 55.3], [12.4, 13.5, 33.6]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %72 = "tf.Const"() {value = dense<[[4.7, 3.8, 5.1], [2.2, 1.3, 3.4]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %82 = "tf.Sub"(%62, %72) {T = f32, device = "CPU:0"} : (tensor<2x3xf32>, tensor<2x3xf32>) -> (tensor<?x?xf32>)
	"tf.DebugPrint"(%82):(tensor<?x?xf32>) -> ()
    %92 = "tf.Neg"(%82) : (tensor<?x?xf32>) -> ( tensor<?x?xf32>)
	"tf.DebugPrint"(%92):(tensor<?x?xf32>) -> ()
 
    // FIXME: enable test require disable `legalizeWithFold`
    //%and_c0 = "tf.Const"() {value = dense<[[0, 1, 1], [1, 1, 0]]> : tensor<2x3xi1>} : () -> tensor<2x3xi1>
    //%and_c1 = "tf.Const"() {value = dense<[[0, 1, 0], [1, 1, 0]]> : tensor<2x3xi1>} : () -> tensor<2x3xi1>
    //%and_c_i1 = "tf.LogicalAnd"(%and_c0, %and_c1) : (tensor<2x3xi1>, tensor<2x3xi1>) -> (tensor<2x3xi1>)
	// "tf.DebugPrint"(%and_c_i1):(tensor<i32>) -> ()

    //%p0 = "tf.Const"() {value = dense<[[false, true, true], [true, true, false]]> : tensor<2x3xi1>} : () -> tensor<2x3xi1>
    //%p00 = "tf.Const"() {value = dense<[[false, true, true], [true, true, false]]> : tensor<2x3xi1>} : () -> tensor<2x3xi1>
    //%select0 = "tf.Select"(%p0, %p0, %p00) : (tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>) -> (tensor<2x3xi1>)
    // FIXME: enable test require disable `legalizeWithFold`
    //%select0 = "tf.Select"(%p0, %p0, %p00) : (tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>) -> (tensor<?x?xi1>)
    // %select1 = "tf.Select"(%select0, %2, %3) : (tensor<?x?xi1>, tensor<?x?xi64>, tensor<?x?xi64>) -> (tensor<?x?xi64>)
	// "tf.DebugPrint"(%select1):(tensor<?x?xi64>) -> ()

    %max0 = "tf.Maximum"(%0, %1) : (tensor<2x3xi64>, tensor<2x3xi64>) -> (tensor<2x3xi64>)
    %max1 = "tf.Maximum"(%2, %3) : (tensor<?x?xi64>, tensor<?x?xi64>) -> (tensor<?x?xi64>)
	"tf.DebugPrint"(%max1):(tensor<?x?xi64>) -> ()
    %const_max0 = "tf.Const"() {value = dense<[[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
    %const_max1 = "tf.Const"() {value = dense<[[34.1, 35.2, 55.3, 11., 24.], [12.4, 13.5, 33.6, 67., 89.9]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
    %const_max2 = "tf.Const"() {value = dense<[[34.1, -35.2, 55.3, -11., 24.], [12.4, 33.5, 33.6, 76., 89.9]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
    %add_max0 = "tf.Add"(%const_max0, %const_max1) {T = f32, device = "CPU"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> (tensor<?x?xf32>)
    %add_max1 = "tf.Add"(%const_max0, %const_max2) {T = f32, device = "CPU"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> (tensor<?x?xf32>)
    %max2 = "tf.Maximum"(%add_max0, %add_max1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?x?xf32>)
	"tf.DebugPrint"(%max2):(tensor<?x?xf32>) -> ()

    %min0 = "tf.Minimum"(%0, %1) : (tensor<2x3xi64>, tensor<2x3xi64>) -> (tensor<2x3xi64>)
    %min1 = "tf.Minimum"(%2, %3) : (tensor<?x?xi64>, tensor<?x?xi64>) -> (tensor<?x?xi64>)
	"tf.DebugPrint"(%min1):(tensor<?x?xi64>) -> ()
    %const_min0 = "tf.Const"() {value = dense<[[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
    %const_min1 = "tf.Const"() {value = dense<[[34.1, 35.2, 55.3, 11., 24.], [12.4, 13.5, 33.6, 67., 89.9]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
    %const_min2 = "tf.Const"() {value = dense<[[34.1, -35.2, 55.3, -11., 24.], [12.4, 33.5, 33.6, 76., 89.9]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
    %add_min0 = "tf.Add"(%const_min0, %const_min1) {T = f32, device = "CPU"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> (tensor<?x?xf32>)
    %add_min1 = "tf.Add"(%const_min0, %const_min2) {T = f32, device = "CPU"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> (tensor<?x?xf32>)
    %min2 = "tf.Minimum"(%add_min0, %add_min1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?x?xf32>)
	"tf.DebugPrint"(%min2):(tensor<?x?xf32>) -> ()

  	%const_rem = xla_hlo.constant dense<[[0, 0, 0], [0, 0, 0]]> : tensor<2x3xi64>
    %const_rem0 = xla_hlo.constant dense<[[34, 35, 55], [12, 13, 33]]> : tensor<2x3xi64>
    %const_rem1 = xla_hlo.constant dense<[[30, 31, 11], [36, 78, 77]]> : tensor<2x3xi64>
    %rem0 = "xla_hlo.remainder"(%const_rem0, %const_rem1) : (tensor<2x3xi64>, tensor<2x3xi64>) -> tensor<2x3xi64>
    %add_rem0 = "xla_hlo.add"(%const_rem, %const_rem0) : (tensor<2x3xi64>, tensor<2x3xi64>) -> tensor<?x?xi64>
    %add_rem1 = "xla_hlo.add"(%const_rem, %const_rem1) : (tensor<2x3xi64>, tensor<2x3xi64>) -> tensor<?x?xi64>
    %rem1 = "xla_hlo.remainder"(%add_rem0, %add_rem1) : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi64>
	"xla_hlo.debug_print"(%rem1) : (tensor<?x?xi64>) -> ()

    %const_rem2 = xla_hlo.constant dense<[[0., 0., 0.], [0., 0., 0.]]> : tensor<2x3xf32>
    %const_rem3 = xla_hlo.constant dense<[[34., 3.5, 5.5], [12., 1.3, 33.]]> : tensor<2x3xf32>
    %const_rem4 = xla_hlo.constant dense<[[3.0, 31., 11.], [36., 78., 7.7]]> : tensor<2x3xf32>
    %add_rem2 = "xla_hlo.add"(%const_rem2, %const_rem3) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<?x?xf32>
    %add_rem3 = "xla_hlo.add"(%const_rem2, %const_rem4) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<?x?xf32>
    %rem2 = "xla_hlo.remainder"(%add_rem2, %add_rem3) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
	"xla_hlo.debug_print"(%rem2) : (tensor<?x?xf32>) -> ()


	%const_cmp0 = xla_hlo.constant dense<[[34, 35, 55], [12, 13, 33]]> : tensor<2x3xi32>
    %const_cmp1 = xla_hlo.constant dense<[[30, 31, 11], [36, 78, 77]]> : tensor<2x3xi32>
    %cmp0 = "xla_hlo.compare"(%const_cmp0, %const_cmp1) {comparison_direction = "LT"} : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>

    %const_cmp = xla_hlo.constant dense<[[0, 0, 0], [0, 0, 0]]> : tensor<2x3xi64>
    %const_cmp2 = xla_hlo.constant dense<[[34, 35, 55], [42, 35, 33]]> : tensor<2x3xi64>
    %const_cmp3 = xla_hlo.constant dense<[[30, 31, 91], [36, 35, 77]]> : tensor<2x3xi64>
    %add_cmp2 = "xla_hlo.add"(%const_cmp, %const_cmp2) : (tensor<2x3xi64>, tensor<2x3xi64>) -> tensor<?x?xi64>
    %add_cmp3 = "xla_hlo.add"(%const_cmp, %const_cmp3) : (tensor<2x3xi64>, tensor<2x3xi64>) -> tensor<?x?xi64>
    %cmp1 = "xla_hlo.compare"(%add_cmp2, %add_cmp3) {comparison_direction = "GT"} : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi1>
	"xla_hlo.debug_print"(%cmp1) : (tensor<?x?xi1>) -> ()
    %cmp2 = "xla_hlo.compare"(%add_cmp2, %add_cmp3) {comparison_direction = "GE"} : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi1>
	"xla_hlo.debug_print"(%cmp2) : (tensor<?x?xi1>) -> ()

    %const_cmp5 = xla_hlo.constant dense<[[0., 0., 0.], [0., 0., 0.]]> : tensor<2x3xf32>
    %const_cmp6 = xla_hlo.constant dense<[[34., 3.5, 5.5], [12., 1.3, 33.]]> : tensor<2x3xf32>
    %const_cmp7 = xla_hlo.constant dense<[[3.0, 31., 11.], [36., 78., 7.7]]> : tensor<2x3xf32>
    %add_cmp5 = "xla_hlo.add"(%const_cmp5, %const_cmp6) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<?x?xf32>
    %add_cmp6 = "xla_hlo.add"(%const_cmp5, %const_cmp7) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<?x?xf32>
    %cmp5 = "xla_hlo.compare"(%add_cmp5, %add_cmp6) {comparison_direction = "LT"} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
	"xla_hlo.debug_print"(%cmp5) : (tensor<?x?xi1>) -> ()

    return
  }
}
