// /opt/circt-prefix/bin/circt-opt --yosys-optimizer="replace-with-optimized-module=true"  test_verilog.mlir
module {

hw.module @CTRL0(in %in0 : i1, in %in1 : i1,in  %in2 : i1, out out0 :i1)
attributes {"#pragma" = "CONTROL_NODE"}
{
    %t20 = comb.xor %in1,%in1 : i1
	%t21 = comb.xor %in0,%in0 : i1
	%t22 = comb.and %t20,%t21 : i1
	%t23 = comb.xor %t22,%t22 : i1
	hw.output %t23 :i1
}

hw.module @CTRL1(in %in0 : i3, out out0 :i8)
attributes {"#pragma" = "CONTROL_NODE"}
{
    %res = SpecHLS.lookUpTable [%in0]:i8 = {0,123,85,26,98,34,4,45}
	hw.output %res :i8
}

hw.module @CTRL2(in %in0 : i1, in %in1 : i1, out out0 :i1)
{
    %t20 = comb.xor %in0,%in1 : i1
	hw.output %t20 :i1
}

hw.module @top(in %a : i1, in %b : i1,in %c : i1,in %d : i1,in %e : i32,out out0 :i1) {
    %t1 = comb.extract %e from 6 : (i32) -> i1
    %t2 = comb.extract %e from 7 : (i32) -> i1
    %t3 = comb.extract %e from 8 : (i32) -> i1
    %t4 = comb.extract %e from 9 : (i32) -> i1
    %t20 = hw.instance "c0" @CTRL0(in0: %t1 : i1, in1: %t2 : i1, in2: %t3 : i1) -> (out0 : i1) {hw.exportPort = @symB}

	hw.output %t20 :i1
}

}