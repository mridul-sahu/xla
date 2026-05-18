// RUN: xla-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

func.func @main(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: s32[4] mulhi
  %0 = "mhlo.mulhi"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}
