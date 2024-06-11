// RUN: spechls-opt --merge-gammas --inline-opt-modules --canonicalize %s | FileCheck %s

module {

hw.module private @test2(in %sel0 : i1, in %sel1 : i1, in %a : i32, in %b : i32, in %c : i32, in %d : i32, out out0 : i32) attributes {"#pragma" = "INLINE"} 
{
  %0 = SpecHLS.gamma @g0 %sel0 ? %a, %b : i32
  %1 = SpecHLS.gamma @g1 %sel0 ? %c, %d : i32
  %2 = SpecHLS.gamma @g2 %sel1 ? %0, %1 : i32
  hw.output %2 : i32 
}

hw.module @test2_tb0(out out0 : i32, out out1 : i32, out out2 : i32, out out3 : i32)
{
//CHECK-LABEL: test2_tb0
//CHECK: hw.output %c11111_i32, %c22222_i32, %c33333_i32, %c44444_i32 : i32, i32, i32, i32
  %in0 = hw.constant 11111 : i32 
  %in1 = hw.constant 22222 : i32 
  %in2 = hw.constant 33333 : i32 
  %in3 = hw.constant 44444 : i32 

  %sel0 = hw.constant 0 : i1
  %sel1 = hw.constant 1 : i1
  
  %unitTest1 = hw.instance "dut1" @test2(sel0: %sel0 : i1, sel1: %sel0 : i1, a: %in0 : i32, b: %in1 : i32, c: %in2 : i32, d: %in3 : i32) -> (out0: i32) 
  %unitTest2 = hw.instance "dut2" @test2(sel0: %sel1 : i1, sel1: %sel0 : i1, a: %in0 : i32, b: %in1 : i32, c: %in2 : i32, d: %in3 : i32) -> (out0: i32) 
  %unitTest3 = hw.instance "dut3" @test2(sel0: %sel0 : i1, sel1: %sel1 : i1, a: %in0 : i32, b: %in1 : i32, c: %in2 : i32, d: %in3 : i32) -> (out0: i32) 
  %unitTest4 = hw.instance "dut4" @test2(sel0: %sel1 : i1, sel1: %sel1 : i1, a: %in0 : i32, b: %in1 : i32, c: %in2 : i32, d: %in3 : i32) -> (out0: i32) 

  hw.output %unitTest1, %unitTest2, %unitTest3, %unitTest4 : i32, i32, i32, i32
}
}
