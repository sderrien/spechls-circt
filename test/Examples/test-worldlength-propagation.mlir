// /opt/circt-prefix/bin/circt-opt --wordlength-proagation  test_verilog.mlir
module {


hw.module @add_u8_imm32(in %in0 : ui8, out out0 :ui8)
{
	%a = SpecHLS.cast %in0:ui8 to i32
    %b = hw.constant 13 : i32
    %add = comb.add %a,%b : i32
	%res = SpecHLS.cast %add:i32 to  ui8
	hw.output %res :ui8
}

hw.module @equ_imm8(in %in0 : ui8, out out0 :i1)
{
	%a = SpecHLS.cast %in0:ui8 to  i32
    %b = hw.constant -13 : i32
    %res = comb.icmp eq %a,%b : i32
	hw.output %res :i1
}












}