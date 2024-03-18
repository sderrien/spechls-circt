// RUN: spechls-opt %s | spechls-opt | FileCheck %s
// CHECK-LABEL:   @top
module {
    hw.module @top(in %in0: memref<16xi32>, out o: i1) {
      %0 = arith.constant 0 : index
      %1 = arith.constant 2 : i32
      %we = arith.constant 1 : i1
      %4 = SpecHLS.sync %we :i1, %in0 : memref<16xi32>, %1: i32
      hw.output %4 : i1
    }
}
