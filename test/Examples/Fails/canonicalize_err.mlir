module {
  hw.module.extern @next(in %lfsr : i32, out out0 : ui32)
  hw.module @SCC_0() {
    %0 = SpecHLS.init @io_state : i32
    %mu = SpecHLS.mu @io_state : %0, %27 : i32
    %1 = SpecHLS.init @__guard : i1
    %mu_0 = SpecHLS.mu @__guard : %1, %16 : i1
    %2 = SpecHLS.init @done : i1
    %mu_1 = SpecHLS.mu @done : %2, %13 : i1
    %3 = SpecHLS.init @i : i32
     %mu_2 = SpecHLS.mu @i : %3, %30 : i32
     %4 = SpecHLS.init @y : i32
     %mu_3 = SpecHLS.mu @y : %4, %33 : i32
     %c128_i32 = hw.constant 128 : i32
     %5 = SpecHLS.cast %mu_3 : i32 to i32
     %call_11.out0 = hw.instance "call_11" @next(lfsr: %5: i32) -> (out0: ui32)
     %6 = SpecHLS.string "%08X\0A" : memref<6xi8>
     %7 = SpecHLS.init @even : i1
     %c1_i32 = hw.constant 1 : i32
     %8 = SpecHLS.cast %c128_i32 : i32 to i32
     %9 = builtin.unrealized_conversion_cast %mu_2 : i32 to i32
     %10 = builtin.unrealized_conversion_cast %8 : i32 to i32
     %11 = comb.icmp sge %9, %10 : i32
     %gamma = SpecHLS.gamma @done %mu_0:i1 ? %mu_1,%11 :i1
     %12 = SpecHLS.cast %gamma : i1 to i1
     %13 = SpecHLS.def @done %12 : i1
     %false = hw.constant false
     %14 = comb.icmp eq %false, %13 : i1
     %15 = SpecHLS.cast %14 : i1 to i1
     %16 = SpecHLS.def @__guard %15 : i1
     %false_4 = hw.constant false
     %17 = comb.icmp eq %false_4, %16 : i1
     %18 = SpecHLS.exit %17 live  %27:i32 ,%16:i1 ,%13:i1 ,%30:i32
     %19 = SpecHLS.cast %7 : i1 to i32
     %20 = SpecHLS.cast %call_11.out0 : ui32 to i32
     %21 = comb.or %19, %20 : i32
     %22 = SpecHLS.ioprintf "%08X\n" (  %21 : i32) from %mu when %mu_0
     %23 = SpecHLS.cast %mu_2 : i32 to i32
     %24 = comb.add %23, %c1_i32 : i32
     %25 = SpecHLS.cast %22 : i32 to i32
     %gamma_5 = SpecHLS.gamma @io_state %mu_0:i1 ? %mu,%25 :i32
     %26 = SpecHLS.cast %gamma_5 : i32 to i32
     %27 = SpecHLS.def @io_state %26 : i32
     %28 = SpecHLS.cast %24 : i32 to i32
     %gamma_6 = SpecHLS.gamma @i %mu_0:i1 ? %mu_2,%28 :i32
     %29 = SpecHLS.cast %gamma_6 : i32 to i32
     %30 = SpecHLS.def @i %29 : i32
     %31 = SpecHLS.cast %call_11.out0 : ui32 to i32
     %gamma_7 = SpecHLS.gamma @y %mu_0:i1 ? %mu_3,%31 :i32
     %32 = SpecHLS.cast %gamma_7 : i32 to i32
     %33 = SpecHLS.def @y %32 : i32
     hw.output
   }
 }