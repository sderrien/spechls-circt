module {
  hw.module @CTRL0(in %in0 : i1, in %in1 : i1, in %in2 : i1, out out0 : i1) attributes {"#pragma" = "CONTROL_NODE"} {
    %0 = comb.xor %in1, %in1 : i1
    %1 = comb.xor %in0, %in0 : i1
    %2 = comb.and %0, %1 : i1
    %3 = comb.xor %2, %2 : i1
    hw.output %3 : i1
  }
  hw.module @CTRL1(in %in0 : i3, out out0 : i8)  {
    %0 = comb.extract %in0 from 0 : (i3) -> i1
    %1 = comb.extract %in0 from 1 : (i3) -> i1
    %2 = comb.extract %in0 from 2 : (i3) -> i1
    %3 = comb.truth_table %0, %1, %2 -> [false, true, true, false, false, false, false, true]
    %4 = comb.truth_table %0, %1, %2 -> [false, true, false, true, true, true, false, false]
    %5 = comb.concat %3, %4 : i1, i1
    %6 = comb.truth_table %0, %1, %2 -> [false, false, true, false, false, false, true, true]
    %7 = comb.concat %5, %6 : i2, i1
    %8 = comb.truth_table %0, %1, %2 -> [false, true, false, true, false, false, false, true]
    %9 = comb.concat %7, %8 : i3, i1
    %10 = comb.truth_table %0, %1, %2 -> [false, true, true, true, false, false, false, false]
    %11 = comb.concat %9, %10 : i4, i1
    %12 = comb.truth_table %0, %1, %2 -> [false, true, false, false, true, true, false, true]
    %13 = comb.concat %11, %12 : i5, i1
    %14 = comb.truth_table %0, %1, %2 -> [false, true, true, false, true, false, false, false]
    %15 = comb.concat %13, %14 : i6, i1
    %16 = comb.truth_table %0, %1, %2 -> [false, false, false, false, false, false, false, false]
    %17 = comb.concat %15, %16 : i7, i1
    hw.output %17 : i8
  }
}