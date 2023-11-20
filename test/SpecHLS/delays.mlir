// RUN: spechls-opt %s | spechls-opt | FileCheck %s
// CHECK-LABEL:   @top
module {
    hw.module @top(in %in0: i32, out o: i32) {
      %1 = arith.constant 2 : i32
      %we = arith.constant 1 : i1
      %2 = SpecHLS.delay %we -> 2: %in0(%1) : i32
      %3 = SpecHLS.mu %in0,%1 : i32
      %4 = SpecHLS.dontCare : i32
      %5 = SpecHLS.exit %we : live %4 : i32
      hw.output %2 : i32
    }
}
