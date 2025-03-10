// RUN: spechls-opt  %s | spechls-opt | FileCheck %s
module {
  hw.module @SCC_0() {
    %0 = SpecHLS.init @io_state : i32
    %1 = SpecHLS.mu @io_state : %0, %25 : i32
    %2 = SpecHLS.init @guard : i1
    %3 = SpecHLS.mu @guard : %2, %19 : i1
    %4 = SpecHLS.init @done : i1
    %5 = SpecHLS.mu @done : %4, %16 : i1
    %true = hw.constant true
    %6 = SpecHLS.string "%08X,%08X\0A" : memref<11xi8>
    %7 = SpecHLS.init @x : i32
    %8 = SpecHLS.init @y : i32
    %9 = comb.add %7, %8 : i32
    %10 = SpecHLS.init @x : i32
    %11 = SpecHLS.init @y : i32
    %12 = comb.sub %10, %11 : i32
    %13 = SpecHLS.ioprintf "%08X,%08X\n" (  %9 : i32, %12 : i32) from %1 when %3
    %14 = SpecHLS.gamma @done %3:i1 ? %5,%true :i1
    %15 = SpecHLS.cast %14 : i1 to i1
    %16 = SpecHLS.def @done %15 : i1
    %false = hw.constant false
    %17 = comb.icmp eq %false, %16 : i1
    %18 = SpecHLS.cast %17 : i1 to i1
    %19 = SpecHLS.def @guard %18 : i1
    %false_0 = hw.constant false
    %20 = comb.icmp eq %false_0, %19 : i1
    %21 = SpecHLS.exit %20 live  %25:i32 ,%19:i1 
    %22 = SpecHLS.cast %13 : i32 to i32
    %23 = SpecHLS.gamma @io_state %3:i1 ? %1,%22 :i32
    %24 = SpecHLS.cast %23 : i32 to i32
    %25 = SpecHLS.def @io_state %24 : i32
    hw.output
  }
}
