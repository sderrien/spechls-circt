// RUN: spechls-opt %s | spechls-opt | FileCheck %s
// CHECK-LABEL:   @top
module {
    hw.module @top(in %in0: ui32, in %in1: ui32,out o: ui32) {
      %2 = arith.addi %in0,%in1 : ui32
      hw.output %2 : ui32
    }
}
