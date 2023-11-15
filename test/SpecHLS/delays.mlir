// RUN: spechls-opt %s | spechls-opt | FileCheck %s
// CHECK-LABEL:   @top
module {
    hw.module @top(%in0: i32) -> (o: i32) {
      %1 = arith.constant 2 : i32
      %we = arith.constant 1 : i1
      %2 = SpecHLS.delay %we -> %in0(%1) : i32
      %3 = SpecHLS.mu %in0,%1 : i32
      %4 = SpecHLS.dontCare : i32
      hw.output %2 : i32
    }
}
