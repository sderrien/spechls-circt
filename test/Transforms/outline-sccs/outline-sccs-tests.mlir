// RUN: spechls-opt --outline-sccs -canonicalize %s | spechls-opt | FileCheck %
//module {
//  hw.module @Counter(in %clock : !seq.clock, in %reset : i1, out count : i8) {
//    %c0_i8 = hw.constant 0 : i8
//    %c1_i8 = hw.constant 1 : i8
//    %0 = SpecHLS.hthread @Counter_SCC_0(%c1_i8, %clock, %reset, %c0_i8 : i8, !seq.clock, i1, i8) -> (i8) {
//    ^bb0(%arg0: i8, %arg1: !seq.clock, %arg2: i1, %arg3: i8):
//      %true = hw.constant true
//      %counter = seq.compreg  %1, %arg1 reset %arg2, %arg3 : i8
//      %1 = comb.add %counter, %arg0 : i8
//      SpecHLS.commit(%counter:i8) when %true
//    }
//    hw.output %0 : i8
//  }
//  hw.module @SCC_0() {
//    %true = hw.constant true
//    %0 = SpecHLS.init @io_state : i32
//    %mu = SpecHLS.mu @io_state : %0, %11 : i32
//    %1 = SpecHLS.init @guard : i1
//    %2 = SpecHLS.init @done : i1
//    %3:2 = SpecHLS.hthread @SCC_0_SCC_0(%2, %true, %1 : i1, i1, i1) -> (i1, i1) {
//    ^bb0(%arg0: i1, %arg1: i1, %arg2: i1):
//      %true_0 = hw.constant true
//      %mu_1 = SpecHLS.mu @guard : %arg2, %14 : i1
//      %mu_2 = SpecHLS.mu @done : %arg0, %12 : i1
//      %gamma_3 = SpecHLS.gamma @done %mu_1:i1 ? %mu_2,%arg1 :i1
//      %12 = SpecHLS.def @done %gamma_3 : i1
//      %13 = comb.xor %12, %arg1 : i1
//      %14 = SpecHLS.def @guard %13 : i1
//      SpecHLS.commit(%mu_1:i1,%14:i1) when %true_0
//    }
//    %4 = SpecHLS.init @x : i32
//    %5 = SpecHLS.init @y : i32
//    %6 = comb.add %4, %5 : i32
//    %7 = SpecHLS.init @x : i32
//    %8 = SpecHLS.init @y : i32
//    %9 = comb.sub %7, %8 : i32
//    %10 = SpecHLS.ioprintf "%08X,%08X\n" (  %6 : i32, %9 : i32) from %mu when %3#0
//    %gamma = SpecHLS.gamma @io_state %3#0:i1 ? %mu,%10 :i32
//    %11 = SpecHLS.def @io_state %gamma : i32
//    hw.output
//  }
//}

module {


  hw.module @Counter(in %clock: !seq.clock, in %reset: i1, out count: i8) {
    %c0_i8 = hw.constant 0 : i8
    %c1_i8 = hw.constant 1 : i8
    %counter = seq.compreg %0, %clock reset %reset, %c0_i8  : i8
    %0 = comb.add %counter, %c1_i8 : i8
      hw.output %counter : i8
  }

  SpecHLS.hkernel "SCC_0" : {
   ^bb0(%arg0: i1, %arg1: i1, %arg2: i1):
    %true = hw.constant true
    %0 = SpecHLS.init @io_state : i32
    %1 = SpecHLS.mu @io_state : %0, %24 : i32
    %2 = SpecHLS.init @guard : i1
    %3 = SpecHLS.mu @guard : %2, %18 : i1
    %4 = SpecHLS.init @done : i1
    %5 = SpecHLS.mu @done : %4, %15 : i1
    %6 = SpecHLS.init @x : i32
    %7 = SpecHLS.init @y : i32
    %8 = comb.add %6, %7 : i32
    %9 = SpecHLS.init @x : i32
    %10 = SpecHLS.init @y : i32
    %11 = comb.sub %9, %10 : i32
    %12 = SpecHLS.ioprintf "%08X,%08X\n" (  %8 : i32, %11 : i32) from %1 when %3
    %13 = SpecHLS.gamma @done %3:i1 ? %5,%true :i1
    %14 = SpecHLS.cast %13 : i1 to i1
    %15 = SpecHLS.def @done %14 : i1
    %16 = comb.xor %15, %true : i1
    %17 = SpecHLS.cast %16 : i1 to i1
    %18 = SpecHLS.def @guard %17 : i1
    %19 = comb.xor %18, %true : i1
    %21 = SpecHLS.cast %12 : i32 to i32
    %22 = SpecHLS.gamma @io_state %3:i1 ? %1,%21 :i32
    %23 = SpecHLS.cast %22 : i32 to i32
    %24 = SpecHLS.def @io_state %23 : i32
    SpecHLS.exit %true
  }
}
