module {


hw.module @SCC_1(
  in %c0 : i2, in %c1 : i1,
  in %x0 : i32, in %x1 : i32, in %x2 : i32, in %x3 : i32, in %x4 : i32, 
  out result : i32)
{
  %0 = SpecHLS.gamma @g0 %c0 ? %x0, %x1, %x2 : i32
  %1 = SpecHLS.gamma @g1 %c0 ? %x1, %x2, %x3, %x4 : i32
  %result = SpecHLS.gamma @g2 %c1 ? %0, %1 : i32
  hw.output %result : i32
}

   hw.module private @DUT(in %mispec : i1, in %guard : i1, out out0: i32) attributes {"#pragma" = "INLINE"} {
      %fast = hw.constant 10 : i32
      %slow = hw.constant 20 : i32
      %_x = hw.constant 30 : i32

      %gamma = SpecHLS.gamma @i %mispec ? %fast,%slow :i32
      %fast1 = SpecHLS.gamma @i %guard ? %gamma,%_x :i32
      hw.output %fast1:i32
   }

    hw.module @TestBench(out out0: i32,out out1: i32,out out2: i32,out out3: i32)   {

        %true = hw.constant 1 : i1
        %false = hw.constant 0 : i1

        %res_0 = hw.instance "dut_0" @DUT(mispec : %false :i1, guard: %false: i1) -> (out0: i32)
        %res_1 = hw.instance "dut_1" @DUT(mispec : %false :i1, guard: %true: i1) -> (out0: i32)
        %res_2 = hw.instance "dut_2" @DUT(mispec : %true  :i1, guard: %false: i1) -> (out0: i32)
        %res_3 = hw.instance "dut_3" @DUT(mispec : %true  :i1, guard: %true: i1) -> (out0: i32)

        hw.output %res_0,%res_1,%res_2,%res_3:i32,i32,i32,i32

   }
}