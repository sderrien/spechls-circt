module {
  hw.module.extern @foo(in %in0 : i32, out out0 : i1)
  hw.module @SCC_0() {
    %false = hw.constant false
    %c32_i32 = hw.constant 32 : i32
    %c1_i32 = hw.constant 1 : i32
    %0 = SpecHLS.init @exit10 : i1
    %1 = SpecHLS.mu @exit10 : %0, %47 : i1
    %2 = SpecHLS.init @guard : i1
    %3 = SpecHLS.mu @guard : %2, %52 : i1
    %4 = SpecHLS.init @guard : i1
    %5 = SpecHLS.mu @guard : %4, %51 : i1
    %6 = SpecHLS.init @i : i32
    %7 = SpecHLS.mu @i : %6, %45 : i32
    %2510 = hw.instance "%10" @foo(in0: %7: i32) -> (out0: i1)
    %8 = SpecHLS.init @x : memref<16xui32>
    %9 = arith.index_cast %7 : i32 to index
    %10 = SpecHLS.read %8 : memref<16xui32>[%9]
    %11 = comb.and %3, %5 : i1
    %12 = comb.and %11, %2510 : i1
    %13 = comb.and %3, %5 : i1
    %14 = comb.icmp eq %false, %2510 : i1
    %15 = comb.and %13, %14 : i1
    %16 = comb.and %3, %5 : i1
    %17 = comb.and %16, %2510 : i1
    %18 = comb.or %15, %17 : i1
    %19 = comb.and %3, %5 : i1
    %20 = comb.icmp eq %false, %2510 : i1
    %21 = comb.and %19, %20 : i1
    %22 = comb.and %3, %5 : i1
    %23 = comb.and %22, %2510 : i1
    %24 = comb.or %21, %23 : i1
    %25 = comb.icmp eq %false, %5 : i1
    %26 = comb.and %3, %25 : i1
    %27 = comb.and %3, %5 : i1
    %28 = comb.icmp eq %false, %2510 : i1
    %29 = comb.and %27, %28 : i1
    %30 = comb.or %26, %29 : i1
    %31 = comb.icmp eq %false, %5 : i1
    %32 = comb.and %3, %31 : i1
    %33 = comb.and %3, %5 : i1
    %34 = comb.icmp eq %false, %2510 : i1
    %35 = comb.and %33, %34 : i1
    %36 = comb.or %32, %35 : i1
    %37 = comb.add %7, %c1_i32 : i32
    %38 = builtin.unrealized_conversion_cast %10 : ui32 to i32
    %SCC_0ctrl_1.out_0 = hw.instance "SCC_0ctrl_1" @SCC_0ctrl_1_opt(in_0: %3: i1, in_1: %5: i1, in_2: %2510: i1) -> (out_0: i1)
    %39 = SpecHLS.gamma @i %SCC_0ctrl_1.out_0 ? %37,%38 :i32
    %40 = comb.concat %36, %24 : i1, i1
    %41 = SpecHLS.lookUpTable [%40 ] :i2= {0,0,1,2 }
    %42 = comb.concat %41, %12 : i2, i1
    %43 = SpecHLS.lookUpTable [%42 ] :i2= {0,0,1,2,2,2,2,2 }
    %SCC_0ctrl_2.out_0 = hw.instance "SCC_0ctrl_2" @SCC_0ctrl_2_opt(in_0: %3: i1, in_1: %5: i1, in_2: %2510: i1) -> (out_0: i2)
    %44 = SpecHLS.gamma @i %SCC_0ctrl_2.out_0 ? %37,%38,%7 :i32
    %45 = SpecHLS.def @i %44 : i32
    %46 = comb.icmp slt %39, %c32_i32 : i32
    %47 = SpecHLS.def @exit10 %1 : i1
    %48 = comb.concat %30, %18 : i1, i1
    %49 = SpecHLS.lookUpTable [%48 ] :i2= {0,0,1,2 }
    %SCC_0ctrl_3.out_0 = hw.instance "SCC_0ctrl_3" @SCC_0ctrl_3_opt(in_0: %3: i1, in_1: %5: i1, in_2: %2510: i1) -> (out_0: i2)
    %50 = SpecHLS.gamma @guard %SCC_0ctrl_3.out_0 ? %1,%46,%5 :i1
    %51 = SpecHLS.def @guard %50 : i1
    %52 = SpecHLS.def @guard %51 : i1
    %53 = comb.icmp eq %false, %52 : i1
    %54 = SpecHLS.exit %53 live  %47:i1 ,%52:i1 ,%51:i1 
    hw.output
  }
  hw.module @SCC_0ctrl_1_opt(in %in_0 : i1, in %in_1 : i1, in %in_2 : i1, out out_0 : i1) {
    %0 = comb.and %in_0, %in_1 : i1
    %1 = comb.and %in_2, %0 : i1
    hw.output %1 : i1
  }
  hw.module @SCC_0ctrl_2_opt(in %in_0 : i1, in %in_1 : i1, in %in_2 : i1, out out_0 : i2) {
    %false = hw.constant false
    %true = hw.constant true
    %0 = comb.concat %true, %false : i1, i1
    hw.output %0 : i2
  }
  hw.module @SCC_0ctrl_3_opt(in %in_0 : i1, in %in_1 : i1, in %in_2 : i1, out out_0 : i2) {
    %0 = comb.xor %in_0 : i1
    %1 = comb.and %in_2, %in_1 : i1
    %2 = comb.and %in_0, %1 : i1
    %3 = comb.concat %0, %2 : i1, i1
    hw.output %3 : i2
  }
}
