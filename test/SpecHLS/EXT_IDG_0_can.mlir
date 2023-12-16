module {
  hw.module @SCC_0() {
    %true = hw.constant true
    %c128_i32 = hw.constant 128 : i32
    %c1_i32 = hw.constant 1 : i32
    %0 = SpecHLS.init "exit11" : i1
    %1 = SpecHLS.mu %0, %24 : i1
    %2 = SpecHLS.init "x" : memref<16xi32>
    %3 = SpecHLS.mu %2, %30 : memref<16xi32>
    %4 = SpecHLS.init "guard" : i1
    %5 = SpecHLS.mu %4, %48 : i1
    %6 = SpecHLS.init "guard" : i1
    %7 = SpecHLS.mu %6, %36 : i1
    %8 = SpecHLS.init "i" : i32
    %9 = SpecHLS.mu %8, %42 : i32
    %10 = comb.add %15, %c1_i32 : i32
    %11 = comb.and %5, %7 : i1
    %12 = arith.index_cast %9 : i32 to index
    %13 = SpecHLS.alpha %11 -> %52[%12], %10 : memref<16xi32>
    %14 = arith.index_cast %9 : i32 to index
    %15 = SpecHLS.read %3 : memref<16xi32>[%14]
    %16 = comb.add %9, %c1_i32 : i32
    %17 = comb.icmp slt %16, %c128_i32 : i32
    %18 = comb.and %5, %7 : i1
    %19 = SpecHLS.gamma %18 ? %1,%17 :i1
    %20 = comb.and %5, %7 : i1
    %21 = SpecHLS.gamma %20 ? %3,%13 :memref<16xi32>
    %22 = comb.and %5, %7 : i1
    %23 = SpecHLS.gamma %22 ? %9,%16 :i32
    %24 = SpecHLS.def "exit11_4" %29 : i1
    %25 = comb.or %26, %28 : i1
    %26 = comb.and %5, %27 : i1
    %27 = comb.xor %7, %true : i1
    %28 = comb.and %5, %7 : i1
    %29 = SpecHLS.gamma %25 ? %1,%19 :i1
    %30 = SpecHLS.def "x_4" %35 : memref<16xi32>
    %31 = comb.or %32, %34 : i1
    %32 = comb.and %5, %33 : i1
    %33 = comb.xor %7, %true : i1
    %34 = comb.and %5, %7 : i1
    %35 = SpecHLS.gamma %31 ? %3,%21 :memref<16xi32>
    %36 = SpecHLS.def "guard_4" %41 : i1
    %37 = comb.or %38, %40 : i1
    %38 = comb.and %5, %39 : i1
    %39 = comb.xor %7, %true : i1
    %40 = comb.and %5, %7 : i1
    %41 = SpecHLS.gamma %37 ? %7,%19 :i1
    %42 = SpecHLS.def "i_5" %47 : i32
    %43 = comb.or %44, %46 : i1
    %44 = comb.and %5, %45 : i1
    %45 = comb.xor %7, %true : i1
    %46 = comb.and %5, %7 : i1
    %47 = SpecHLS.gamma %43 ? %9,%23 :i32
    %48 = SpecHLS.def "guard_3" %36 : i1
    %49 = comb.xor %48, %true : i1
    %50 = SpecHLS.delay %true -> %49 by 4 : i1
    %51 = SpecHLS.exit %50
    %52 = SpecHLS.sync %3 : memref<16xi32>, %15 : i32
    hw.output
  }
}

