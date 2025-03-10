// RUN: spechls-opt --merge-luts --inline-opt-modules --canonicalize %s | spechls-opt | FileCheck %s
// CHECK:module {
// CHECK:  hw.module @testbench(out out_0 : i4, out out_1 : i4, out out_2 : i4, out out_3 : i4, out out_4 : i4, out out_5 : i4, out out_6 : i4, out out_7 : i4) {
// CHECK:    %c7_i4 = hw.constant 7 : i4
// CHECK:    %c5_i4 = hw.constant 5 : i4
// CHECK:    %c-8_i4 = hw.constant -8 : i4
// CHECK:    %c1_i4 = hw.constant 1 : i4
// CHECK:    %c6_i4 = hw.constant 6 : i4
// CHECK:    %c4_i4 = hw.constant 4 : i4
// CHECK:    hw.output %c7_i4, %c5_i4, %c-8_i4, %c1_i4, %c-8_i4, %c6_i4, %c4_i4, %c7_i4 : i4, i4, i4, i4, i4, i4, i4, i4
// CHECK:  }
// CHECK:}

module {

  hw.module @dut(in %in_0 : i3, out res : i4) attributes {"#pragma" = "INLINE"}  {
    %0 = comb.extract %in_0 from 0 : (i3) -> i2
    %1 = comb.extract %in_0 from 2 : (i3) -> i1
    %innerLUT = SpecHLS.lookUpTable [%0 : i2] :i2= {0,1,2,3}
    %addr = comb.concat %innerLUT, %1 : i2, i1
    %outerLUT = SpecHLS.lookUpTable [%addr : i3] :i4= {0,1,2,3,4,5,6,7}
    hw.output %outerLUT:i4
  }

 hw.module @testbench(
    out out_0 : i4,
    out out_1 : i4,
    out out_2 : i4,
    out out_3 : i4,
    out out_4 : i4,
    out out_5 : i4,
    out out_6 : i4,
    out out_7 : i4
 ) {

	%inner0 = hw.constant 0 :i3
	%inner1 = hw.constant 1 :i3
	%inner2 = hw.constant 2 :i3
	%inner3 = hw.constant 3 :i3
	%inner4 = hw.constant 4 :i3
	%inner5 = hw.constant 5 :i3
	%inner6 = hw.constant 6 :i3
	%inner7 = hw.constant 7 :i3

	%res0 = hw.instance "lut_0" @dut(in_0 : %inner0 :i3) -> (res: i4)
	%res1 = hw.instance "lut_1" @dut(in_0 : %inner1 :i3) -> (res: i4)
	%res2 = hw.instance "lut_2" @dut(in_0 : %inner2 :i3) -> (res: i4)
	%res3 = hw.instance "lut_3" @dut(in_0 : %inner3 :i3) -> (res: i4)
	%res4 = hw.instance "lut_4" @dut(in_0 : %inner4 :i3) -> (res: i4)
	%res5 = hw.instance "lut_5" @dut(in_0 : %inner5 :i3) -> (res: i4)
	%res6 = hw.instance "lut_6" @dut(in_0 : %inner6 :i3) -> (res: i4)
	%res7 = hw.instance "lut_7" @dut(in_0 : %inner7 :i3) -> (res: i4)

	hw.output
	  %res0,  %res1,  %res2,   %res3,
	  %res4,  %res5,   %res6,  %res7 :
	  i4 , i4 , i4 , i4 ,
	  i4 , i4 ,i4 , i4

	  }


}
