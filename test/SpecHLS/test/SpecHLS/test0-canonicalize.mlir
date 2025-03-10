// RUN: spechls-opt --canonicalize %s | spechls-opt | FileCheck %s
module {
  hw.module @SCC_0() {
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
    %20 = SpecHLS.exit %19 live  %24:i32 ,%18:i1 
    %21 = SpecHLS.cast %12 : i32 to i32
    %22 = SpecHLS.gamma @io_state %3:i1 ? %1,%21 :i32
    %23 = SpecHLS.cast %22 : i32 to i32
    %24 = SpecHLS.def @io_state %23 : i32
    hw.output
  }
}
