// /opt/circt-prefix/bin/circt-opt --export-verilog test_verilog.mlir

hw.module @SCC_0(in %in0 : i32, in %in1 : i32, out out0 :i32) {

	%t17 = comb.add %in0,%in1 : i32
	// node 0 at 16
	%t18 = hw.constant 12 : i32
	// node 0 at 17
	%t20 = comb.xor %t17,%t18 : i32
	// node 0 at 18
	hw.output %t20 :i32
	// node 0 at 23

}
