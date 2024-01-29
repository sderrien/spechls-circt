// RUN: spechls-opt --export-vitis-hls %s |  FileCheck %s
hw.module @SCC_0(in %in_0 : i32, in %in_1 : i32, in %in_2 : i1, in %in_3 : i1, in %mem : memref<16xi32>,  out out_0 : i32) {

	%one = hw.constant 1 : i32
	%t17 = comb.add %in_0,%one : i32
	%t23 = comb.and %t17,%in_1 : i32
	%t24 = comb.xor %t23,%in_0 : i32
    %t1 = comb.extract %t24 from 0 : (i32) -> i1
	%t27 = comb.mux %t1,%t24,%t17 : i32

    %gctrl = comb.extract %t24 from 0 : (i32) -> i2

    %addr4 = comb.extract %t24 from 0 : (i32) -> i4

	%g_out = SpecHLS.gamma @toto %gctrl ? %t24, %t17, %in_0, %t27 :i32
    %lut_in = comb.extract %t24 from 2 : (i32) -> i3
    %lout = SpecHLS.lookUpTable [%lut_in]:i32 = {0,3,5,6,8,3334,4564,45}

    %we = comb.extract %t24 from 5 : (i32) -> i1
    %delay_out = SpecHLS.delay %we -> %t27 by 2 (%t27) : i32
    %mu_out = SpecHLS.mu @adfag : %t24,%t27 : i32
   // %printf_out = SpecHLS.print "toto %d" : %t24 : i32
    %5 = SpecHLS.exit %we


    hw.output %t27:i32
}
