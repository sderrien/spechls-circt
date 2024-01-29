// RUN: spechls-opt %s | spechls-opt | FileCheck %s
// CHECK-LABEL:   @top
module {
    hw.module @top(in %in0: memref<16xi32>, out o: memref<16xi32>) {
      %0 = arith.constant 0 : index
      %1 = arith.constant 2 : i32
      %we = arith.constant 1 : i1
      %2 = SpecHLS.alpha @tab %we -> %in0[%0], %1 : memref<16xi32>
      %3 = SpecHLS.alpha @tab %we -> %2[%0], %1 : memref<16xi32>
      hw.output %3 : memref<16xi32>
    }
}
