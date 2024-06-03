// RUN: spechls-opt --factor-gamma-inputs --inline-opt-modules --canonicalize %s | spechls-opt | FileCheck %s
// CHECK:module {
// CHECK:  hw.module @test1_tb(out out0 : i32) {
// CHECK:    %c33333_i32 = hw.constant 33333 : i32
// CHECK:    hw.output %c33333_i32 : i32
// CHECK:  }
// CHECK:}

module{
  hw.module private @test1(in %sel : i2, in %a : i32, in %b : i32, in %c : i8, in %d : i24, out out0 : i32) attributes {"#pragma" = "INLINE"}
  {
    %1 = comb.concat %c, %d : i8, i24
    %2 = comb.concat %d, %c : i24, i8
    %3 = comb.add %a, %b : i32
    %4 = comb.sub %b, %a : i32
    %5 = SpecHLS.gamma @x %sel ? %1, %2, %3, %4 : i32
    hw.output %5 : i32
  }

  
  hw.module @test1_tb(out out0 :i32) {

    %in0 = hw.constant 11111 :i32
    %in1 = hw.constant 22222 :i32
    %in2 = hw.constant 10    :i8
    %in3 = hw.constant 44444 :i24

    %sel0 = hw.constant 0 :i2
    %sel1 = hw.constant 1 :i2
    %sel2 = hw.constant 2 :i2
    %sel3 = hw.constant 3 :i2

    %unitTest0 = hw.instance "dut0" @test1(sel : %sel2 :i2, a: %in0: i32, b: %in1: i32,c: %in2: i8,d: %in3: i24) -> (out0: i32)

    hw.output %unitTest0: i32
  }

}
