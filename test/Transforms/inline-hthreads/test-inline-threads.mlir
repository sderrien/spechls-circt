module {
  hw.module @Counter(in %clock : !seq.clock, in %reset : i1, out count : i8) {
    %c0_i8 = hw.constant 0 : i8
    %c1_i8 = hw.constant 1 : i8
    %0 = SpecHLS.hthread @Counter_SCC_0(%c1_i8, %clock, %reset, %c0_i8 : i8, !seq.clock, i1, i8) -> (i8) {
    ^bb0(%arg0: i8, %arg1: !seq.clock, %arg2: i1, %arg3: i8):
      %true = hw.constant true
      %counter = seq.compreg  %1, %arg1 reset %arg2, %arg3 : i8
      %1 = comb.add %counter, %arg0 : i8
      SpecHLS.commit(%counter:i8) when %true
    }
    hw.output %0 : i8
  }
  hw.module @SCC_0() {
    %true = hw.constant true
    %0 = SpecHLS.init @io_state : i32
    %mu = SpecHLS.mu @io_state : %0, %13 : i32
    %1 = SpecHLS.init @guard : i1
    %2 = SpecHLS.init @done : i1
    %3:2 = SpecHLS.hthread @SCC_0_SCC_0(%2, %true, %1 : i1, i1, i1) -> (i1, i1) {
    ^bb0(%arg0: i1, %arg1: i1, %arg2: i1):
      %true_0 = hw.constant true
      %mu_1 = SpecHLS.mu @guard : %arg2, %16 : i1
      %mu_2 = SpecHLS.mu @done : %arg0, %14 : i1
      %gamma = SpecHLS.gamma @done %mu_1:i1 ? %mu_2,%arg1 :i1
      %14 = SpecHLS.def @done %gamma : i1
      %15 = comb.xor %14, %arg1 : i1
      %16 = SpecHLS.def @guard %15 : i1
      SpecHLS.commit(%mu_1:i1,%16:i1) when %true_0
    }
    %4 = SpecHLS.init @x : i32
    %5 = SpecHLS.init @y : i32
    %6 = comb.add %4, %5 : i32
    %7 = SpecHLS.init @x : i32
    %8 = SpecHLS.init @y : i32
    %9 = comb.sub %7, %8 : i32
    %10 = SpecHLS.ioprintf "%08X,%08X\n" (  %6 : i32, %9 : i32) from %mu when %3#0
    %11 = comb.xor %3#1, %true : i1
    %12 = SpecHLS.exit %11 live  %13:i32
    %13 = SpecHLS.hthread @SCC_0_SCC_0(%3#0, %6, %9, %0 : i1, i32, i32, i32) -> (i32) {
    ^bb0(%arg0: i1, %arg1: i32, %arg2: i32, %arg3: i32):
      %true_0 = hw.constant true
      %mu_1 = SpecHLS.mu @io_state : %arg3, %15 : i32
      %14 = SpecHLS.ioprintf "%08X,%08X\n" (  %arg1 : i32, %arg2 : i32) from %mu_1 when %arg0
      %gamma = SpecHLS.gamma @io_state %arg0:i1 ? %mu_1,%14 :i32
      %15 = SpecHLS.def @io_state %gamma : i32
      SpecHLS.commit(%15:i32) when %true_0
    }
    hw.output
  }
}
