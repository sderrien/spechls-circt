// RUN: spechls-opt --factor-gamma-inputs --inline-opt-modules --canonicalize %s | spechls-opt | FileCheck %s
// CHECK:module {
// CHECK:  hw.module @test1_tb(out out0 : i32) {
// CHECK:    %c43346_i32 = hw.constant 43346 : i32
// CHECK:    hw.output %c43346_i32 : i32
// CHECK:  }
// CHECK:}


module {

   hw.module private @test1(in %sel : i3, in %a : i32, in %b: i32,in %c: i32,in %d:i32,in %e: i32,in %f:i32,in %g:i32, out out0 :i32) attributes {"#pragma" = "INLINE"}
   {
       %1 = comb.xor %a,%b :i32
       %2 = comb.add %c,%d :i32
       %3 = comb.add %e,%f :i32
       %4 = comb.sub %b,%e :i32
       %5 = comb.sub %a,%c :i32
       %6 = comb.xor %b,%a :i32
       %7 = comb.add %a,%b :i32
       %8 = comb.sub %e,%d :i32
       %10 = SpecHLS.gamma @x %sel ?  %1, %2, %3, %4, %5, %6, %7, %8 :i32
       hw.output %10 : i32
   }

   hw.module @test1_tb(out out0 :i32) {

          %in0 = hw.constant 11111 :i32
          %in1 = hw.constant 22222 :i32
          %in2 = hw.constant 33333 :i32
          %in3 = hw.constant 44444 :i32
          %in4 = hw.constant 55555 :i32
          %in5 = hw.constant 66666 :i32
          %in6 = hw.constant 77777 :i32
          %in7 = hw.constant 88888 :i32

          %sel0 = hw.constant 0 :i3
          %sel1 = hw.constant 1 :i3
          %sel2 = hw.constant 2 :i3
          %sel3 = hw.constant 3 :i3
          %sel4 = hw.constant 4 :i3
          %sel5 = hw.constant 5 :i3
          %sel6 = hw.constant 6 :i3
          %sel7 = hw.constant 7 :i3

          %unitTest0 = hw.instance "dut0" @test1(sel : %sel0 :i3, a: %in0: i32, b: %in2: i32,c: %in3: i32,d: %in4: i32,e: %in5: i32,f: %in6: i32,g: %in7: i32) -> (out0: i32)

          hw.output %unitTest0: i32
      }
}

