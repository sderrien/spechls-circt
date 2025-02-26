module {
  hw.module @SCC_0() {
    %c6_i32 = hw.constant 6 : i32
    %true = hw.constant true
    %c4_i32 = hw.constant 4 : i32
    %c1_i32 = hw.constant 1 : i32
    %0 = SpecHLS.init @io_state : i32
    %mu = SpecHLS.mu @io_state : %0, %21 : i32
    %1 = SpecHLS.init @__guard : i1
    %mu_0 = SpecHLS.mu @__guard : %1, %27 : i1
    %2 = SpecHLS.init @x : i32
    %mu_1 = SpecHLS.mu @x : %2, %15 : i32
    %3 = SpecHLS.init @i : i32
    %mu_2 = SpecHLS.mu @i : %3, %24 : i32
    %4 = SpecHLS.init @c1 : i32
    %5 = comb.icmp slt %mu_2, %c4_i32 : i32
    %6 = comb.and %mu_0, %5 : i1
    %7 = comb.xor %5, %true : i1
    %8 = comb.xor %5, %true : i1
    %9 = comb.and %mu_0, %8 : i1
    %10 = comb.or %6, %9 : i1
    %11 = SpecHLS.cast %mu_1 : i32 to i32
    %12 = comb.add %11, %c1_i32 : i32
    %13 = SpecHLS.cast %12 : i32 to i32
    %gamma = SpecHLS.gamma @x %7:i1 ? %13,%4 :i32
    %gamma_3 = SpecHLS.gamma @x %10:i1 ? %mu_1,%gamma :i32
    %14 = SpecHLS.cast %gamma_3 : i32 to i32
    %15 = SpecHLS.def @x %14 : i32
    %16 = SpecHLS.cast %mu_2 : i32 to i32
    %17 = comb.add %16, %c1_i32 : i32
    %18 = SpecHLS.ioprintf "%d %d\n" (  %17 : i32, %gamma : i32) from %mu when %10
    %19 = SpecHLS.cast %18 : i32 to i32
    %gamma_4 = SpecHLS.gamma @io_state %10:i1 ? %mu,%19 :i32
    %20 = SpecHLS.cast %gamma_4 : i32 to i32
    %21 = SpecHLS.def @io_state %20 : i32
    %22 = SpecHLS.cast %17 : i32 to i32
    %gamma_5 = SpecHLS.gamma @i %10:i1 ? %mu_2,%22 :i32
    %23 = SpecHLS.cast %gamma_5 : i32 to i32
    %24 = SpecHLS.def @i %23 : i32
    %25 = comb.icmp slt %24, %c6_i32 : i32
    %26 = SpecHLS.cast %25 : i1 to i1
    %27 = SpecHLS.def @__guard %26 : i1
    %28 = comb.xor %27, %true : i1
    %29 = SpecHLS.exit %28 live  %21:i32 ,%27:i1 ,%15:i32
    hw.output
  }
}
