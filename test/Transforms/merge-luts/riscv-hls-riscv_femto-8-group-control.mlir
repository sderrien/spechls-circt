module {
   hw.module @SCC_0_ctrl_0(in %in_0 : i1, in %in_1 : i8, in %in_2 : i8, out out_0 : i2) attributes {"#pragma" = "CONTROL_NODE"} {
    %c99_i8 = hw.constant 99 : i8
    %c1_i8 = hw.constant 1 : i8
    %true = hw.constant true
    %0 = comb.xor %in_0, %true : i1
    %1 = comb.icmp eq %in_2, %c1_i8 : i8
    %2 = comb.xor %1, %true : i1
    %3 = comb.icmp eq %in_1, %c99_i8 : i8
    %4 = comb.and %3, %1 : i1
    %5 = comb.and %4, %in_0 : i1
    %6 = comb.and %4, %0 : i1
    %7 = comb.or %5, %6 : i1
    %8 = comb.and %3, %2 : i1
    %9 = comb.or %7, %8 : i1
    %10 = comb.concat %9, %7 : i1, i1
    %LUT = SpecHLS.lookUpTable [%10 : i2] :i2= {0,0,1,2 }
    %11 = comb.concat %LUT, %5 : i2, i1
    %LUT_0 = SpecHLS.lookUpTable [%11 : i3] :i3= {0,0,1,1,2,3,4,4 }
    %LUT_1 = SpecHLS.lookUpTable [%LUT_0 : i3] :i2= {0,0,0,1,2,2,2,2 }
    hw.output %LUT_1 : i2
  }
  hw.module @SCC_0_ctrl_1(in %in_0 : i3, in %in_1 : i1, in %in_2 : i8, in %in_3 : i8, in %in_4 : i8, in %in_5 : i8, in %in_6 : i1, in %in_7 : i8, in %in_8 : i8, out out_0 : i1) attributes {"#pragma" = "CONTROL_NODE"} {
    %c99_i8 = hw.constant 99 : i8
    %c51_i8 = hw.constant 51 : i8
    %c19_i8 = hw.constant 19 : i8
    %c1_i8 = hw.constant 1 : i8
    %c0_i8 = hw.constant 0 : i8
    %true = hw.constant true
    %0 = comb.xor %in_1, %true : i1
    %1 = comb.icmp eq %in_3, %c0_i8 : i8
    %2 = comb.icmp eq %in_5, %c0_i8 : i8
    %3 = comb.xor %2, %true : i1
    %4 = comb.xor %in_6, %true : i1
    %5 = comb.icmp eq %in_8, %c1_i8 : i8
    %6 = comb.icmp eq %in_2, %c19_i8 : i8
    %7 = comb.and %6, %1 : i1
    %8 = comb.and %7, %in_1 : i1
    %9 = comb.and %7, %0 : i1
    %10 = comb.or %8, %9 : i1
    %11 = comb.icmp eq %in_4, %c51_i8 : i8
    %12 = comb.and %11, %1 : i1
    %13 = comb.and %12, %2 : i1
    %14 = comb.and %13, %in_1 : i1
    %15 = comb.and %13, %0 : i1
    %16 = comb.or %14, %15 : i1
    %17 = comb.and %12, %3 : i1
    %18 = comb.or %17, %16 : i1
    %19 = comb.icmp eq %in_7, %c99_i8 : i8
    %20 = comb.and %19, %5 : i1
    %21 = comb.and %20, %in_6 : i1
    %22 = comb.and %20, %4 : i1
    %23 = comb.or %21, %22 : i1
    %24 = comb.concat %in_0, %10 : i3, i1
    %LUT = SpecHLS.lookUpTable [%24 : i4] :i3= {0,0,1,2,3,3,4,4,5,5,6,6,6,6,6,6 }
    %25 = comb.concat %LUT, %18 : i3, i1
    %LUT_0 = SpecHLS.lookUpTable [%25 : i4] :i3= {0,0,1,1,2,2,3,4,5,5,6,6,7,7,7,7 }
    %26 = comb.concat %LUT_0, %16 : i3, i1
    %LUT_1 = SpecHLS.lookUpTable [%26 : i4] :i4= {0,0,1,1,2,2,3,3,4,5,6,6,7,7,8,8 }
    %27 = comb.concat %LUT_1, %23 : i4, i1
    %LUT_2 = SpecHLS.lookUpTable [%27 : i5] :i4= {0,0,1,1,2,2,3,3,4,4,5,5,6,7,8,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9 }
    %LUT_3 = SpecHLS.lookUpTable [%LUT_2 : i4] :i3= {0,0,1,0,0,2,0,3,4,5,5,5,5,5,5,5 }
    %LUT_4 = SpecHLS.lookUpTable [%LUT_3 : i3] :i1= {0,1,1,1,1,1,1,1 }
    hw.output %LUT_4 : i1
  }
  hw.module @SCC_0_ctrl_2(in %in_0 : i1, in %in_1 : i1, in %in_2 : i1, in %in_3 : i1, in %in_4 : i8, in %in_5 : i8, in %in_6 : i8, in %in_7 : i8, in %in_8 : i8, in %in_9 : i1, in %in_10 : i8, in %in_11 : i8, in %in_12 : i8, out out_0 : i1) attributes {"#pragma" = "CONTROL_NODE"} {
    %c115_i8 = hw.constant 115 : i8
    %c99_i8 = hw.constant 99 : i8
    %c51_i8 = hw.constant 51 : i8
    %c19_i8 = hw.constant 19 : i8
    %c1_i8 = hw.constant 1 : i8
    %c5_i8 = hw.constant 5 : i8
    %c0_i8 = hw.constant 0 : i8
    %true = hw.constant true
    %0 = comb.xor %in_2, %true : i1
    %1 = comb.icmp eq %in_5, %c0_i8 : i8
    %2 = comb.xor %1, %true : i1
    %3 = comb.icmp eq %in_8, %c0_i8 : i8
    %4 = comb.xor %3, %true : i1
    %5 = comb.icmp eq %in_7, %c5_i8 : i8
    %6 = comb.or %1, %5 : i1
    %7 = comb.xor %6, %true : i1
    %8 = comb.xor %in_9, %true : i1
    %9 = comb.icmp eq %in_11, %c1_i8 : i8
    %10 = comb.xor %9, %true : i1
    %11 = comb.icmp eq %in_4, %c19_i8 : i8
    %12 = comb.and %in_3, %11 : i1
    %13 = comb.and %12, %1 : i1
    %14 = comb.and %13, %in_2 : i1
    %15 = comb.and %13, %0 : i1
    %16 = comb.and %12, %2 : i1
    %17 = comb.icmp eq %in_6, %c51_i8 : i8
    %18 = comb.and %in_3, %17 : i1
    %19 = comb.and %18, %1 : i1
    %20 = comb.and %19, %3 : i1
    %21 = comb.and %20, %in_2 : i1
    %22 = comb.and %20, %0 : i1
    %23 = comb.and %19, %4 : i1
    %24 = comb.and %18, %5 : i1
    %25 = comb.and %18, %7 : i1
    %26 = comb.icmp eq %in_10, %c99_i8 : i8
    %27 = comb.and %in_3, %26 : i1
    %28 = comb.and %27, %9 : i1
    %29 = comb.and %28, %in_9 : i1
    %30 = comb.and %28, %8 : i1
    %31 = comb.and %27, %10 : i1
    %32 = comb.icmp eq %in_12, %c115_i8 : i8
    %33 = comb.and %in_3, %32 : i1
    %34 = comb.or %11, %17, %26, %32 : i1
    %35 = comb.xor %34, %true : i1
    %36 = comb.and %in_3, %35 : i1
    %37 = comb.or %14, %15, %16, %25, %23, %21, %22, %24, %29, %30, %31, %33, %36 : i1
    %38 = comb.and %37, %in_1 : i1
    %39 = comb.xor %in_1, %true : i1
    %40 = comb.and %37, %39 : i1
    %41 = comb.and %38, %in_0 : i1
    %42 = comb.xor %in_0, %true : i1
    %43 = comb.and %38, %42 : i1
    %44 = comb.or %41, %43, %40 : i1
    %45 = comb.concat %44, %39 : i1, i1
    %LUT = SpecHLS.lookUpTable [%45 : i2] :i2= {0,0,1,2 }
    %46 = comb.concat %LUT, %41 : i2, i1
    %LUT_0 = SpecHLS.lookUpTable [%46 : i3] :i3= {0,0,1,2,3,3,4,4 }
    %47 = comb.concat %LUT_0, %36 : i3, i1
    %LUT_1 = SpecHLS.lookUpTable [%47 : i4] :i3= {0,0,1,2,3,3,4,4,5,5,5,5,5,5,5,5 }
    %LUT_2 = SpecHLS.lookUpTable [%LUT_1 : i3] :i3= {0,0,1,2,3,4,4,4 }
    %LUT_3 = SpecHLS.lookUpTable [%LUT_2 : i3] :i1= {0,1,1,1,1,1,1,1 }
    hw.output %LUT_3 : i1
  }
  hw.module @SCC_0_ctrl_3(in %in_0 : i1, in %in_1 : i1, in %in_2 : i8, in %in_3 : i8, in %in_4 : i8, in %in_5 : i8, in %in_6 : i8, in %in_7 : i1, in %in_8 : i8, in %in_9 : i8, in %in_10 : i8, in %in_11 : i1, in %in_12 : i1, out out_0 : i2) attributes {"#pragma" = "CONTROL_NODE"} {
    %c115_i8 = hw.constant 115 : i8
    %c99_i8 = hw.constant 99 : i8
    %c51_i8 = hw.constant 51 : i8
    %c19_i8 = hw.constant 19 : i8
    %c1_i8 = hw.constant 1 : i8
    %c5_i8 = hw.constant 5 : i8
    %c0_i8 = hw.constant 0 : i8
    %true = hw.constant true
    %0 = comb.xor %in_0, %true : i1
    %1 = comb.icmp eq %in_3, %c0_i8 : i8
    %2 = comb.xor %1, %true : i1
    %3 = comb.icmp eq %in_6, %c0_i8 : i8
    %4 = comb.xor %3, %true : i1
    %5 = comb.icmp eq %in_5, %c5_i8 : i8
    %6 = comb.or %1, %5 : i1
    %7 = comb.xor %6, %true : i1
    %8 = comb.xor %in_7, %true : i1
    %9 = comb.icmp eq %in_9, %c1_i8 : i8
    %10 = comb.xor %9, %true : i1
    %11 = comb.icmp eq %in_2, %c19_i8 : i8
    %12 = comb.and %in_1, %11 : i1
    %13 = comb.and %12, %1 : i1
    %14 = comb.and %13, %in_0 : i1
    %15 = comb.and %13, %0 : i1
    %16 = comb.and %12, %2 : i1
    %17 = comb.icmp eq %in_4, %c51_i8 : i8
    %18 = comb.and %in_1, %17 : i1
    %19 = comb.and %18, %1 : i1
    %20 = comb.and %19, %3 : i1
    %21 = comb.and %20, %in_0 : i1
    %22 = comb.and %20, %0 : i1
    %23 = comb.and %19, %4 : i1
    %24 = comb.and %18, %5 : i1
    %25 = comb.and %18, %7 : i1
    %26 = comb.icmp eq %in_8, %c99_i8 : i8
    %27 = comb.and %in_1, %26 : i1
    %28 = comb.and %27, %9 : i1
    %29 = comb.and %28, %in_7 : i1
    %30 = comb.and %28, %8 : i1
    %31 = comb.and %27, %10 : i1
    %32 = comb.icmp eq %in_10, %c115_i8 : i8
    %33 = comb.and %in_1, %32 : i1
    %34 = comb.or %11, %17, %26, %32 : i1
    %35 = comb.xor %34, %true : i1
    %36 = comb.and %in_1, %35 : i1
    %37 = comb.or %14, %15, %16, %25, %23, %21, %22, %24, %29, %30, %31, %33, %36 : i1
    %38 = comb.xor %in_11, %true : i1
    %39 = comb.and %in_11, %in_12 : i1
    %40 = comb.xor %in_12, %true : i1
    %41 = comb.and %in_11, %40 : i1
    %42 = comb.or %39, %41, %38 : i1
    %43 = comb.and %37, %42 : i1
    %44 = comb.concat %43, %33 : i1, i1
    %LUT = SpecHLS.lookUpTable [%44 : i2] :i2= {0,0,1,2 }
    %LUT_0 = SpecHLS.lookUpTable [%LUT : i2] :i2= {0,0,1,2 }
    hw.output %LUT_0 : i2
  }
  hw.module @SCC_0_ctrl_4(in %in_0 : i1, in %in_1 : i1, in %in_2 : i1, in %in_3 : i1, in %in_4 : i8, in %in_5 : i8, in %in_6 : i8, in %in_7 : i8, in %in_8 : i8, in %in_9 : i1, in %in_10 : i8, in %in_11 : i8, in %in_12 : i8, out out_0 : i2) attributes {"#pragma" = "CONTROL_NODE"} {
    %c115_i8 = hw.constant 115 : i8
    %c99_i8 = hw.constant 99 : i8
    %c51_i8 = hw.constant 51 : i8
    %c19_i8 = hw.constant 19 : i8
    %c1_i8 = hw.constant 1 : i8
    %c5_i8 = hw.constant 5 : i8
    %c0_i8 = hw.constant 0 : i8
    %true = hw.constant true
    %0 = comb.xor %in_2, %true : i1
    %1 = comb.icmp eq %in_5, %c0_i8 : i8
    %2 = comb.xor %1, %true : i1
    %3 = comb.icmp eq %in_8, %c0_i8 : i8
    %4 = comb.xor %3, %true : i1
    %5 = comb.icmp eq %in_7, %c5_i8 : i8
    %6 = comb.or %1, %5 : i1
    %7 = comb.xor %6, %true : i1
    %8 = comb.xor %in_9, %true : i1
    %9 = comb.icmp eq %in_11, %c1_i8 : i8
    %10 = comb.xor %9, %true : i1
    %11 = comb.icmp eq %in_4, %c19_i8 : i8
    %12 = comb.and %in_3, %11 : i1
    %13 = comb.and %12, %1 : i1
    %14 = comb.and %13, %in_2 : i1
    %15 = comb.and %13, %0 : i1
    %16 = comb.and %12, %2 : i1
    %17 = comb.icmp eq %in_6, %c51_i8 : i8
    %18 = comb.and %in_3, %17 : i1
    %19 = comb.and %18, %1 : i1
    %20 = comb.and %19, %3 : i1
    %21 = comb.and %20, %in_2 : i1
    %22 = comb.and %20, %0 : i1
    %23 = comb.and %19, %4 : i1
    %24 = comb.and %18, %5 : i1
    %25 = comb.and %18, %7 : i1
    %26 = comb.icmp eq %in_10, %c99_i8 : i8
    %27 = comb.and %in_3, %26 : i1
    %28 = comb.and %27, %9 : i1
    %29 = comb.and %28, %in_9 : i1
    %30 = comb.and %28, %8 : i1
    %31 = comb.and %27, %10 : i1
    %32 = comb.icmp eq %in_12, %c115_i8 : i8
    %33 = comb.and %in_3, %32 : i1
    %34 = comb.or %11, %17, %26, %32 : i1
    %35 = comb.xor %34, %true : i1
    %36 = comb.and %in_3, %35 : i1
    %37 = comb.or %14, %15, %16, %25, %23, %21, %22, %24, %29, %30, %31, %33, %36 : i1
    %38 = comb.and %37, %in_1 : i1
    %39 = comb.xor %in_1, %true : i1
    %40 = comb.and %37, %39 : i1
    %41 = comb.and %38, %in_0 : i1
    %42 = comb.xor %in_0, %true : i1
    %43 = comb.and %38, %42 : i1
    %44 = comb.or %41, %43, %40 : i1
    %45 = comb.concat %44, %38 : i1, i1
    %LUT = SpecHLS.lookUpTable [%45 : i2] :i2= {0,0,1,2 }
    %LUT_0 = SpecHLS.lookUpTable [%LUT : i2] :i2= {0,0,1,2 }
    hw.output %LUT_0 : i2
  }
  hw.module @SCC_0_ctrl_5(in %in_0 : i1, in %in_1 : i1, in %in_2 : i1, in %in_3 : i1, in %in_4 : i8, in %in_5 : i8, in %in_6 : i8, in %in_7 : i8, in %in_8 : i8, in %in_9 : i1, in %in_10 : i8, in %in_11 : i8, in %in_12 : i8, out out_0 : i2) attributes {"#pragma" = "CONTROL_NODE"} {
    %c115_i8 = hw.constant 115 : i8
    %c99_i8 = hw.constant 99 : i8
    %c51_i8 = hw.constant 51 : i8
    %c19_i8 = hw.constant 19 : i8
    %c1_i8 = hw.constant 1 : i8
    %c5_i8 = hw.constant 5 : i8
    %c0_i8 = hw.constant 0 : i8
    %true = hw.constant true
    %0 = comb.xor %in_2, %true : i1
    %1 = comb.icmp eq %in_5, %c0_i8 : i8
    %2 = comb.xor %1, %true : i1
    %3 = comb.icmp eq %in_8, %c0_i8 : i8
    %4 = comb.xor %3, %true : i1
    %5 = comb.icmp eq %in_7, %c5_i8 : i8
    %6 = comb.or %1, %5 : i1
    %7 = comb.xor %6, %true : i1
    %8 = comb.xor %in_9, %true : i1
    %9 = comb.icmp eq %in_11, %c1_i8 : i8
    %10 = comb.xor %9, %true : i1
    %11 = comb.icmp eq %in_4, %c19_i8 : i8
    %12 = comb.and %in_3, %11 : i1
    %13 = comb.and %12, %1 : i1
    %14 = comb.and %13, %in_2 : i1
    %15 = comb.and %13, %0 : i1
    %16 = comb.and %12, %2 : i1
    %17 = comb.icmp eq %in_6, %c51_i8 : i8
    %18 = comb.and %in_3, %17 : i1
    %19 = comb.and %18, %1 : i1
    %20 = comb.and %19, %3 : i1
    %21 = comb.and %20, %in_2 : i1
    %22 = comb.and %20, %0 : i1
    %23 = comb.and %19, %4 : i1
    %24 = comb.and %18, %5 : i1
    %25 = comb.and %18, %7 : i1
    %26 = comb.icmp eq %in_10, %c99_i8 : i8
    %27 = comb.and %26, %9 : i1
    %28 = comb.and %27, %in_9 : i1
    %29 = comb.and %27, %8 : i1
    %30 = comb.or %28, %29 : i1
    %31 = comb.and %26, %10 : i1
    %32 = comb.or %30, %31 : i1
    %33 = comb.and %in_3, %26 : i1
    %34 = comb.and %33, %9 : i1
    %35 = comb.and %34, %in_9 : i1
    %36 = comb.and %34, %8 : i1
    %37 = comb.and %33, %10 : i1
    %38 = comb.icmp eq %in_12, %c115_i8 : i8
    %39 = comb.and %in_3, %38 : i1
    %40 = comb.or %11, %17, %26, %38 : i1
    %41 = comb.xor %40, %true : i1
    %42 = comb.and %in_3, %41 : i1
    %43 = comb.or %14, %15, %16, %25, %23, %21, %22, %24, %35, %36, %37, %39, %42 : i1
    %44 = comb.and %43, %in_1 : i1
    %45 = comb.xor %in_1, %true : i1
    %46 = comb.and %43, %45 : i1
    %47 = comb.and %44, %in_0 : i1
    %48 = comb.xor %in_0, %true : i1
    %49 = comb.and %44, %48 : i1
    %50 = comb.or %47, %49, %46 : i1
    %51 = comb.concat %50, %44 : i1, i1
    %LUT = SpecHLS.lookUpTable [%51 : i2] :i2= {0,0,1,2 }
    %52 = comb.concat %LUT, %32 : i2, i1
    %LUT_0 = SpecHLS.lookUpTable [%52 : i3] :i3= {0,0,1,1,2,3,4,4 }
    %53 = comb.concat %LUT_0, %30 : i3, i1
    %LUT_1 = SpecHLS.lookUpTable [%53 : i4] :i3= {0,0,1,1,2,2,3,4,5,5,5,5,5,5,5,5 }
    %54 = comb.concat %LUT_1, %28 : i3, i1
    %LUT_2 = SpecHLS.lookUpTable [%54 : i4] :i3= {0,0,1,1,2,2,3,3,4,5,6,6,6,6,6,6 }
    %LUT_3 = SpecHLS.lookUpTable [%LUT_2 : i3] :i3= {0,0,1,2,3,4,5,5 }
    %LUT_4 = SpecHLS.lookUpTable [%LUT_3 : i3] :i2= {0,1,1,1,2,3,3,3 }
    hw.output %LUT_4 : i2
  }
  hw.module @SCC_0_ctrl_6(in %in_0 : i1, in %in_1 : i1, in %in_2 : i1, in %in_3 : i1, in %in_4 : i8, in %in_5 : i8, in %in_6 : i8, in %in_7 : i8, in %in_8 : i8, in %in_9 : i1, in %in_10 : i8, in %in_11 : i8, in %in_12 : i8, out out_0 : i3) attributes {"#pragma" = "CONTROL_NODE"} {
    %c115_i8 = hw.constant 115 : i8
    %c99_i8 = hw.constant 99 : i8
    %c51_i8 = hw.constant 51 : i8
    %c19_i8 = hw.constant 19 : i8
    %c1_i8 = hw.constant 1 : i8
    %c5_i8 = hw.constant 5 : i8
    %c0_i8 = hw.constant 0 : i8
    %true = hw.constant true
    %0 = comb.xor %in_2, %true : i1
    %1 = comb.icmp eq %in_5, %c0_i8 : i8
    %2 = comb.xor %1, %true : i1
    %3 = comb.icmp eq %in_8, %c0_i8 : i8
    %4 = comb.xor %3, %true : i1
    %5 = comb.icmp eq %in_7, %c5_i8 : i8
    %6 = comb.or %1, %5 : i1
    %7 = comb.xor %6, %true : i1
    %8 = comb.xor %in_9, %true : i1
    %9 = comb.icmp eq %in_11, %c1_i8 : i8
    %10 = comb.xor %9, %true : i1
    %11 = comb.icmp eq %in_4, %c19_i8 : i8
    %12 = comb.and %in_3, %11 : i1
    %13 = comb.and %12, %1 : i1
    %14 = comb.and %13, %in_2 : i1
    %15 = comb.and %13, %0 : i1
    %16 = comb.and %12, %2 : i1
    %17 = comb.icmp eq %in_6, %c51_i8 : i8
    %18 = comb.and %in_3, %17 : i1
    %19 = comb.and %18, %1 : i1
    %20 = comb.and %19, %3 : i1
    %21 = comb.and %20, %in_2 : i1
    %22 = comb.and %20, %0 : i1
    %23 = comb.and %19, %4 : i1
    %24 = comb.and %18, %5 : i1
    %25 = comb.and %18, %7 : i1
    %26 = comb.icmp eq %in_10, %c99_i8 : i8
    %27 = comb.and %in_3, %26 : i1
    %28 = comb.and %27, %9 : i1
    %29 = comb.and %28, %in_9 : i1
    %30 = comb.and %28, %8 : i1
    %31 = comb.and %27, %10 : i1
    %32 = comb.icmp eq %in_12, %c115_i8 : i8
    %33 = comb.and %in_3, %32 : i1
    %34 = comb.or %11, %17, %26, %32 : i1
    %35 = comb.xor %34, %true : i1
    %36 = comb.and %in_3, %35 : i1
    %37 = comb.or %14, %15, %16, %25, %23, %21, %22, %24, %29, %30, %31, %33, %36 : i1
    %38 = comb.and %37, %in_1 : i1
    %39 = comb.xor %in_1, %true : i1
    %40 = comb.and %37, %39 : i1
    %41 = comb.and %38, %in_0 : i1
    %42 = comb.xor %in_0, %true : i1
    %43 = comb.and %38, %42 : i1
    %44 = comb.or %41, %43, %40 : i1
    %45 = comb.concat %44, %39 : i1, i1
    %LUT = SpecHLS.lookUpTable [%45 : i2] :i2= {0,0,1,2 }
    %46 = comb.concat %LUT, %41 : i2, i1
    %LUT_0 = SpecHLS.lookUpTable [%46 : i3] :i3= {0,0,1,2,3,3,4,4 }
    hw.output %LUT_0 : i3
  }
  hw.module @SCC_0_ctrl_7(in %in_0 : i2, in %in_1 : i1, in %in_2 : i1, in %in_3 : i8, in %in_4 : i8, in %in_5 : i8, in %in_6 : i8, in %in_7 : i8, in %in_8 : i1, in %in_9 : i8, in %in_10 : i8, in %in_11 : i8, in %in_12 : i1, in %in_13 : i1, out out_0 : i1) attributes {"#pragma" = "CONTROL_NODE"} {
    %c115_i8 = hw.constant 115 : i8
    %c99_i8 = hw.constant 99 : i8
    %c51_i8 = hw.constant 51 : i8
    %c19_i8 = hw.constant 19 : i8
    %c1_i8 = hw.constant 1 : i8
    %c5_i8 = hw.constant 5 : i8
    %c0_i8 = hw.constant 0 : i8
    %true = hw.constant true
    %0 = comb.xor %in_1, %true : i1
    %1 = comb.icmp eq %in_4, %c0_i8 : i8
    %2 = comb.xor %1, %true : i1
    %3 = comb.icmp eq %in_7, %c0_i8 : i8
    %4 = comb.xor %3, %true : i1
    %5 = comb.icmp eq %in_6, %c5_i8 : i8
    %6 = comb.or %1, %5 : i1
    %7 = comb.xor %6, %true : i1
    %8 = comb.xor %in_8, %true : i1
    %9 = comb.icmp eq %in_10, %c1_i8 : i8
    %10 = comb.xor %9, %true : i1
    %11 = comb.icmp eq %in_3, %c19_i8 : i8
    %12 = comb.and %in_2, %11 : i1
    %13 = comb.and %12, %1 : i1
    %14 = comb.and %13, %in_1 : i1
    %15 = comb.and %13, %0 : i1
    %16 = comb.or %14, %15 : i1
    %17 = comb.and %12, %2 : i1
    %18 = comb.icmp eq %in_5, %c51_i8 : i8
    %19 = comb.and %in_2, %18 : i1
    %20 = comb.and %19, %1 : i1
    %21 = comb.and %20, %3 : i1
    %22 = comb.and %21, %in_1 : i1
    %23 = comb.and %21, %0 : i1
    %24 = comb.or %22, %23 : i1
    %25 = comb.and %20, %4 : i1
    %26 = comb.or %25, %24 : i1
    %27 = comb.and %19, %5 : i1
    %28 = comb.and %19, %7 : i1
    %29 = comb.icmp eq %in_9, %c99_i8 : i8
    %30 = comb.and %in_2, %29 : i1
    %31 = comb.and %30, %9 : i1
    %32 = comb.and %31, %in_8 : i1
    %33 = comb.and %31, %8 : i1
    %34 = comb.and %30, %10 : i1
    %35 = comb.icmp eq %in_11, %c115_i8 : i8
    %36 = comb.and %in_2, %35 : i1
    %37 = comb.or %11, %18, %29, %35 : i1
    %38 = comb.xor %37, %true : i1
    %39 = comb.and %in_2, %38 : i1
    %40 = comb.or %16, %17, %28, %26, %27, %32, %33, %34, %36, %39 : i1
    %41 = comb.xor %in_12, %true : i1
    %42 = comb.and %in_12, %in_13 : i1
    %43 = comb.xor %in_13, %true : i1
    %44 = comb.and %in_12, %43 : i1
    %45 = comb.or %42, %44, %41 : i1
    %46 = comb.and %40, %45 : i1
    %47 = comb.concat %46, %in_0 : i1, i2
    %LUT = SpecHLS.lookUpTable [%47 : i3] :i3= {0,0,0,0,1,2,3,4 }
    %48 = comb.concat %LUT, %16 : i3, i1
    %LUT_0 = SpecHLS.lookUpTable [%48 : i4] :i3= {0,0,1,1,2,3,4,4,5,5,5,5,5,5,5,5 }
    %49 = comb.concat %LUT_0, %14 : i3, i1
    %LUT_1 = SpecHLS.lookUpTable [%49 : i4] :i3= {0,0,1,1,2,2,3,4,5,5,6,6,6,6,6,6 }
    %50 = comb.concat %LUT_1, %26 : i3, i1
    %LUT_2 = SpecHLS.lookUpTable [%50 : i4] :i3= {0,0,1,1,2,2,3,3,4,4,5,6,7,7,7,7 }
    %51 = comb.concat %LUT_2, %24 : i3, i1
    %LUT_3 = SpecHLS.lookUpTable [%51 : i4] :i4= {0,0,1,1,2,2,3,3,4,4,5,5,6,7,8,8 }
    %52 = comb.concat %LUT_3, %22 : i4, i1
    %LUT_4 = SpecHLS.lookUpTable [%52 : i5] :i4= {0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9 }
    %LUT_5 = SpecHLS.lookUpTable [%LUT_4 : i4] :i2= {0,0,0,0,1,0,0,0,2,3,3,3,3,3,3,3 }
    %LUT_6 = SpecHLS.lookUpTable [%LUT_5 : i2] :i1= {0,0,1,0 }
    %LUT_7 = SpecHLS.lookUpTable [%LUT_6 : i1] :i1= {0,1 }
    hw.output %LUT_7 : i1
  }
  hw.module @SCC_0_ctrl_8(in %in_0 : i2, in %in_1 : i1, in %in_2 : i1, in %in_3 : i8, in %in_4 : i8, in %in_5 : i8, in %in_6 : i8, in %in_7 : i8, in %in_8 : i1, in %in_9 : i8, in %in_10 : i8, in %in_11 : i8, in %in_12 : i1, in %in_13 : i1, out out_0 : i1) attributes {"#pragma" = "CONTROL_NODE"} {
    %c115_i8 = hw.constant 115 : i8
    %c99_i8 = hw.constant 99 : i8
    %c51_i8 = hw.constant 51 : i8
    %c19_i8 = hw.constant 19 : i8
    %c1_i8 = hw.constant 1 : i8
    %c5_i8 = hw.constant 5 : i8
    %c0_i8 = hw.constant 0 : i8
    %true = hw.constant true
    %0 = comb.xor %in_1, %true : i1
    %1 = comb.icmp eq %in_4, %c0_i8 : i8
    %2 = comb.xor %1, %true : i1
    %3 = comb.icmp eq %in_7, %c0_i8 : i8
    %4 = comb.xor %3, %true : i1
    %5 = comb.icmp eq %in_6, %c5_i8 : i8
    %6 = comb.or %1, %5 : i1
    %7 = comb.xor %6, %true : i1
    %8 = comb.xor %in_8, %true : i1
    %9 = comb.icmp eq %in_10, %c1_i8 : i8
    %10 = comb.xor %9, %true : i1
    %11 = comb.icmp eq %in_3, %c19_i8 : i8
    %12 = comb.and %in_2, %11 : i1
    %13 = comb.and %12, %1 : i1
    %14 = comb.and %13, %in_1 : i1
    %15 = comb.and %13, %0 : i1
    %16 = comb.or %14, %15 : i1
    %17 = comb.and %12, %2 : i1
    %18 = comb.icmp eq %in_5, %c51_i8 : i8
    %19 = comb.and %in_2, %18 : i1
    %20 = comb.and %19, %1 : i1
    %21 = comb.and %20, %3 : i1
    %22 = comb.and %21, %in_1 : i1
    %23 = comb.and %21, %0 : i1
    %24 = comb.or %22, %23 : i1
    %25 = comb.and %20, %4 : i1
    %26 = comb.or %25, %24 : i1
    %27 = comb.and %19, %5 : i1
    %28 = comb.and %19, %7 : i1
    %29 = comb.icmp eq %in_9, %c99_i8 : i8
    %30 = comb.and %in_2, %29 : i1
    %31 = comb.and %30, %9 : i1
    %32 = comb.and %31, %in_8 : i1
    %33 = comb.and %31, %8 : i1
    %34 = comb.and %30, %10 : i1
    %35 = comb.icmp eq %in_11, %c115_i8 : i8
    %36 = comb.and %in_2, %35 : i1
    %37 = comb.or %11, %18, %29, %35 : i1
    %38 = comb.xor %37, %true : i1
    %39 = comb.and %in_2, %38 : i1
    %40 = comb.or %16, %17, %28, %26, %27, %32, %33, %34, %36, %39 : i1
    %41 = comb.xor %in_12, %true : i1
    %42 = comb.and %in_12, %in_13 : i1
    %43 = comb.xor %in_13, %true : i1
    %44 = comb.and %in_12, %43 : i1
    %45 = comb.or %42, %44, %41 : i1
    %46 = comb.and %40, %45 : i1
    %47 = comb.concat %46, %in_0 : i1, i2
    %LUT = SpecHLS.lookUpTable [%47 : i3] :i3= {0,0,0,0,1,2,3,4 }
    %48 = comb.concat %LUT, %16 : i3, i1
    %LUT_0 = SpecHLS.lookUpTable [%48 : i4] :i3= {0,0,1,1,2,3,4,4,5,5,5,5,5,5,5,5 }
    %49 = comb.concat %LUT_0, %14 : i3, i1
    %LUT_1 = SpecHLS.lookUpTable [%49 : i4] :i3= {0,0,1,1,2,2,3,4,5,5,6,6,6,6,6,6 }
    %50 = comb.concat %LUT_1, %26 : i3, i1
    %LUT_2 = SpecHLS.lookUpTable [%50 : i4] :i3= {0,0,1,1,2,2,3,3,4,4,5,6,7,7,7,7 }
    %51 = comb.concat %LUT_2, %24 : i3, i1
    %LUT_3 = SpecHLS.lookUpTable [%51 : i4] :i4= {0,0,1,1,2,2,3,3,4,4,5,5,6,7,8,8 }
    %52 = comb.concat %LUT_3, %22 : i4, i1
    %LUT_4 = SpecHLS.lookUpTable [%52 : i5] :i4= {0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9 }
    %LUT_5 = SpecHLS.lookUpTable [%LUT_4 : i4] :i2= {0,0,0,0,1,0,0,0,2,3,3,3,3,3,3,3 }
    %LUT_6 = SpecHLS.lookUpTable [%LUT_5 : i2] :i1= {0,0,1,0 }
    %LUT_7 = SpecHLS.lookUpTable [%LUT_6 : i1] :i1= {0,1 }
    hw.output %LUT_7 : i1
  }
  hw.module @SCC_0_ctrl_9(in %in_0 : i2, in %in_1 : i1, in %in_2 : i1, in %in_3 : i8, in %in_4 : i8, in %in_5 : i8, in %in_6 : i8, in %in_7 : i8, in %in_8 : i1, in %in_9 : i8, in %in_10 : i8, in %in_11 : i8, in %in_12 : i1, in %in_13 : i1, out out_0 : i1) attributes {"#pragma" = "CONTROL_NODE"} {
    %c115_i8 = hw.constant 115 : i8
    %c99_i8 = hw.constant 99 : i8
    %c51_i8 = hw.constant 51 : i8
    %c19_i8 = hw.constant 19 : i8
    %c1_i8 = hw.constant 1 : i8
    %c5_i8 = hw.constant 5 : i8
    %c0_i8 = hw.constant 0 : i8
    %true = hw.constant true
    %0 = comb.xor %in_1, %true : i1
    %1 = comb.icmp eq %in_4, %c0_i8 : i8
    %2 = comb.xor %1, %true : i1
    %3 = comb.icmp eq %in_7, %c0_i8 : i8
    %4 = comb.xor %3, %true : i1
    %5 = comb.icmp eq %in_6, %c5_i8 : i8
    %6 = comb.or %1, %5 : i1
    %7 = comb.xor %6, %true : i1
    %8 = comb.xor %in_8, %true : i1
    %9 = comb.icmp eq %in_10, %c1_i8 : i8
    %10 = comb.xor %9, %true : i1
    %11 = comb.icmp eq %in_3, %c19_i8 : i8
    %12 = comb.and %in_2, %11 : i1
    %13 = comb.and %12, %1 : i1
    %14 = comb.and %13, %in_1 : i1
    %15 = comb.and %13, %0 : i1
    %16 = comb.or %14, %15 : i1
    %17 = comb.and %12, %2 : i1
    %18 = comb.icmp eq %in_5, %c51_i8 : i8
    %19 = comb.and %in_2, %18 : i1
    %20 = comb.and %19, %1 : i1
    %21 = comb.and %20, %3 : i1
    %22 = comb.and %21, %in_1 : i1
    %23 = comb.and %21, %0 : i1
    %24 = comb.or %22, %23 : i1
    %25 = comb.and %20, %4 : i1
    %26 = comb.or %25, %24 : i1
    %27 = comb.and %19, %5 : i1
    %28 = comb.and %19, %7 : i1
    %29 = comb.icmp eq %in_9, %c99_i8 : i8
    %30 = comb.and %in_2, %29 : i1
    %31 = comb.and %30, %9 : i1
    %32 = comb.and %31, %in_8 : i1
    %33 = comb.and %31, %8 : i1
    %34 = comb.and %30, %10 : i1
    %35 = comb.icmp eq %in_11, %c115_i8 : i8
    %36 = comb.and %in_2, %35 : i1
    %37 = comb.or %11, %18, %29, %35 : i1
    %38 = comb.xor %37, %true : i1
    %39 = comb.and %in_2, %38 : i1
    %40 = comb.or %16, %17, %28, %26, %27, %32, %33, %34, %36, %39 : i1
    %41 = comb.xor %in_12, %true : i1
    %42 = comb.and %in_12, %in_13 : i1
    %43 = comb.xor %in_13, %true : i1
    %44 = comb.and %in_12, %43 : i1
    %45 = comb.or %42, %44, %41 : i1
    %46 = comb.and %40, %45 : i1
    %47 = comb.concat %46, %in_0 : i1, i2
    %LUT = SpecHLS.lookUpTable [%47 : i3] :i3= {0,0,0,0,1,2,3,4 }
    %48 = comb.concat %LUT, %16 : i3, i1
    %LUT_0 = SpecHLS.lookUpTable [%48 : i4] :i3= {0,0,1,1,2,3,4,4,5,5,5,5,5,5,5,5 }
    %49 = comb.concat %LUT_0, %14 : i3, i1
    %LUT_1 = SpecHLS.lookUpTable [%49 : i4] :i3= {0,0,1,1,2,2,3,4,5,5,6,6,6,6,6,6 }
    %50 = comb.concat %LUT_1, %26 : i3, i1
    %LUT_2 = SpecHLS.lookUpTable [%50 : i4] :i3= {0,0,1,1,2,2,3,3,4,4,5,6,7,7,7,7 }
    %51 = comb.concat %LUT_2, %24 : i3, i1
    %LUT_3 = SpecHLS.lookUpTable [%51 : i4] :i4= {0,0,1,1,2,2,3,3,4,4,5,5,6,7,8,8 }
    %52 = comb.concat %LUT_3, %22 : i4, i1
    %LUT_4 = SpecHLS.lookUpTable [%52 : i5] :i4= {0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9 }
    %LUT_5 = SpecHLS.lookUpTable [%LUT_4 : i4] :i2= {0,0,0,0,1,0,0,0,2,3,3,3,3,3,3,3 }
    %LUT_6 = SpecHLS.lookUpTable [%LUT_5 : i2] :i1= {0,0,1,0 }
    %LUT_7 = SpecHLS.lookUpTable [%LUT_6 : i1] :i1= {0,1 }
    hw.output %LUT_7 : i1
  }
  hw.module @SCC_0_ctrl_10(in %in_0 : i2, in %in_1 : i1, in %in_2 : i1, in %in_3 : i8, in %in_4 : i8, in %in_5 : i8, in %in_6 : i8, in %in_7 : i8, in %in_8 : i1, in %in_9 : i8, in %in_10 : i8, in %in_11 : i8, in %in_12 : i1, in %in_13 : i1, out out_0 : i1) attributes {"#pragma" = "CONTROL_NODE"} {
    %c115_i8 = hw.constant 115 : i8
    %c99_i8 = hw.constant 99 : i8
    %c51_i8 = hw.constant 51 : i8
    %c19_i8 = hw.constant 19 : i8
    %c1_i8 = hw.constant 1 : i8
    %c5_i8 = hw.constant 5 : i8
    %c0_i8 = hw.constant 0 : i8
    %true = hw.constant true
    %0 = comb.xor %in_1, %true : i1
    %1 = comb.icmp eq %in_4, %c0_i8 : i8
    %2 = comb.xor %1, %true : i1
    %3 = comb.icmp eq %in_7, %c0_i8 : i8
    %4 = comb.xor %3, %true : i1
    %5 = comb.icmp eq %in_6, %c5_i8 : i8
    %6 = comb.or %1, %5 : i1
    %7 = comb.xor %6, %true : i1
    %8 = comb.xor %in_8, %true : i1
    %9 = comb.icmp eq %in_10, %c1_i8 : i8
    %10 = comb.xor %9, %true : i1
    %11 = comb.icmp eq %in_3, %c19_i8 : i8
    %12 = comb.and %in_2, %11 : i1
    %13 = comb.and %12, %1 : i1
    %14 = comb.and %13, %in_1 : i1
    %15 = comb.and %13, %0 : i1
    %16 = comb.or %14, %15 : i1
    %17 = comb.and %12, %2 : i1
    %18 = comb.icmp eq %in_5, %c51_i8 : i8
    %19 = comb.and %in_2, %18 : i1
    %20 = comb.and %19, %1 : i1
    %21 = comb.and %20, %3 : i1
    %22 = comb.and %21, %in_1 : i1
    %23 = comb.and %21, %0 : i1
    %24 = comb.or %22, %23 : i1
    %25 = comb.and %20, %4 : i1
    %26 = comb.or %25, %24 : i1
    %27 = comb.and %19, %5 : i1
    %28 = comb.and %19, %7 : i1
    %29 = comb.icmp eq %in_9, %c99_i8 : i8
    %30 = comb.and %in_2, %29 : i1
    %31 = comb.and %30, %9 : i1
    %32 = comb.and %31, %in_8 : i1
    %33 = comb.and %31, %8 : i1
    %34 = comb.and %30, %10 : i1
    %35 = comb.icmp eq %in_11, %c115_i8 : i8
    %36 = comb.and %in_2, %35 : i1
    %37 = comb.or %11, %18, %29, %35 : i1
    %38 = comb.xor %37, %true : i1
    %39 = comb.and %in_2, %38 : i1
    %40 = comb.or %16, %17, %28, %26, %27, %32, %33, %34, %36, %39 : i1
    %41 = comb.xor %in_12, %true : i1
    %42 = comb.and %in_12, %in_13 : i1
    %43 = comb.xor %in_13, %true : i1
    %44 = comb.and %in_12, %43 : i1
    %45 = comb.or %42, %44, %41 : i1
    %46 = comb.and %40, %45 : i1
    %47 = comb.concat %46, %in_0 : i1, i2
    %LUT = SpecHLS.lookUpTable [%47 : i3] :i3= {0,0,0,0,1,2,3,4 }
    %48 = comb.concat %LUT, %16 : i3, i1
    %LUT_0 = SpecHLS.lookUpTable [%48 : i4] :i3= {0,0,1,1,2,3,4,4,5,5,5,5,5,5,5,5 }
    %49 = comb.concat %LUT_0, %14 : i3, i1
    %LUT_1 = SpecHLS.lookUpTable [%49 : i4] :i3= {0,0,1,1,2,2,3,4,5,5,6,6,6,6,6,6 }
    %50 = comb.concat %LUT_1, %26 : i3, i1
    %LUT_2 = SpecHLS.lookUpTable [%50 : i4] :i3= {0,0,1,1,2,2,3,3,4,4,5,6,7,7,7,7 }
    %51 = comb.concat %LUT_2, %24 : i3, i1
    %LUT_3 = SpecHLS.lookUpTable [%51 : i4] :i4= {0,0,1,1,2,2,3,3,4,4,5,5,6,7,8,8 }
    %52 = comb.concat %LUT_3, %22 : i4, i1
    %LUT_4 = SpecHLS.lookUpTable [%52 : i5] :i4= {0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9 }
    %LUT_5 = SpecHLS.lookUpTable [%LUT_4 : i4] :i2= {0,0,0,0,1,0,0,0,2,3,3,3,3,3,3,3 }
    %LUT_6 = SpecHLS.lookUpTable [%LUT_5 : i2] :i1= {0,0,1,0 }
    %LUT_7 = SpecHLS.lookUpTable [%LUT_6 : i1] :i1= {0,1 }
    hw.output %LUT_7 : i1
  }
  hw.module @SCC_0_ctrl_11(in %in_0 : i2, in %in_1 : i1, in %in_2 : i1, in %in_3 : i8, in %in_4 : i8, in %in_5 : i8, in %in_6 : i8, in %in_7 : i8, in %in_8 : i1, in %in_9 : i8, in %in_10 : i8, in %in_11 : i8, in %in_12 : i1, in %in_13 : i1, out out_0 : i1) attributes {"#pragma" = "CONTROL_NODE"} {
    %c115_i8 = hw.constant 115 : i8
    %c99_i8 = hw.constant 99 : i8
    %c51_i8 = hw.constant 51 : i8
    %c19_i8 = hw.constant 19 : i8
    %c1_i8 = hw.constant 1 : i8
    %c5_i8 = hw.constant 5 : i8
    %c0_i8 = hw.constant 0 : i8
    %true = hw.constant true
    %0 = comb.xor %in_1, %true : i1
    %1 = comb.icmp eq %in_4, %c0_i8 : i8
    %2 = comb.xor %1, %true : i1
    %3 = comb.icmp eq %in_7, %c0_i8 : i8
    %4 = comb.xor %3, %true : i1
    %5 = comb.icmp eq %in_6, %c5_i8 : i8
    %6 = comb.or %1, %5 : i1
    %7 = comb.xor %6, %true : i1
    %8 = comb.xor %in_8, %true : i1
    %9 = comb.icmp eq %in_10, %c1_i8 : i8
    %10 = comb.xor %9, %true : i1
    %11 = comb.icmp eq %in_3, %c19_i8 : i8
    %12 = comb.and %in_2, %11 : i1
    %13 = comb.and %12, %1 : i1
    %14 = comb.and %13, %in_1 : i1
    %15 = comb.and %13, %0 : i1
    %16 = comb.or %14, %15 : i1
    %17 = comb.and %12, %2 : i1
    %18 = comb.icmp eq %in_5, %c51_i8 : i8
    %19 = comb.and %in_2, %18 : i1
    %20 = comb.and %19, %1 : i1
    %21 = comb.and %20, %3 : i1
    %22 = comb.and %21, %in_1 : i1
    %23 = comb.and %21, %0 : i1
    %24 = comb.or %22, %23 : i1
    %25 = comb.and %20, %4 : i1
    %26 = comb.or %25, %24 : i1
    %27 = comb.and %19, %5 : i1
    %28 = comb.and %19, %7 : i1
    %29 = comb.icmp eq %in_9, %c99_i8 : i8
    %30 = comb.and %in_2, %29 : i1
    %31 = comb.and %30, %9 : i1
    %32 = comb.and %31, %in_8 : i1
    %33 = comb.and %31, %8 : i1
    %34 = comb.and %30, %10 : i1
    %35 = comb.icmp eq %in_11, %c115_i8 : i8
    %36 = comb.and %in_2, %35 : i1
    %37 = comb.or %11, %18, %29, %35 : i1
    %38 = comb.xor %37, %true : i1
    %39 = comb.and %in_2, %38 : i1
    %40 = comb.or %16, %17, %28, %26, %27, %32, %33, %34, %36, %39 : i1
    %41 = comb.xor %in_12, %true : i1
    %42 = comb.and %in_12, %in_13 : i1
    %43 = comb.xor %in_13, %true : i1
    %44 = comb.and %in_12, %43 : i1
    %45 = comb.or %42, %44, %41 : i1
    %46 = comb.and %40, %45 : i1
    %47 = comb.concat %46, %in_0 : i1, i2
    %LUT = SpecHLS.lookUpTable [%47 : i3] :i3= {0,0,0,0,1,2,3,4 }
    %48 = comb.concat %LUT, %16 : i3, i1
    %LUT_0 = SpecHLS.lookUpTable [%48 : i4] :i3= {0,0,1,1,2,3,4,4,5,5,5,5,5,5,5,5 }
    %49 = comb.concat %LUT_0, %14 : i3, i1
    %LUT_1 = SpecHLS.lookUpTable [%49 : i4] :i3= {0,0,1,1,2,2,3,4,5,5,6,6,6,6,6,6 }
    %50 = comb.concat %LUT_1, %26 : i3, i1
    %LUT_2 = SpecHLS.lookUpTable [%50 : i4] :i3= {0,0,1,1,2,2,3,3,4,4,5,6,7,7,7,7 }
    %51 = comb.concat %LUT_2, %24 : i3, i1
    %LUT_3 = SpecHLS.lookUpTable [%51 : i4] :i4= {0,0,1,1,2,2,3,3,4,4,5,5,6,7,8,8 }
    %52 = comb.concat %LUT_3, %22 : i4, i1
    %LUT_4 = SpecHLS.lookUpTable [%52 : i5] :i4= {0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9 }
    %LUT_5 = SpecHLS.lookUpTable [%LUT_4 : i4] :i2= {0,0,0,0,1,0,0,0,2,3,3,3,3,3,3,3 }
    %LUT_6 = SpecHLS.lookUpTable [%LUT_5 : i2] :i1= {0,0,1,0 }
    %LUT_7 = SpecHLS.lookUpTable [%LUT_6 : i1] :i1= {0,1 }
    hw.output %LUT_7 : i1
  }
  hw.module @SCC_0_ctrl_12(in %in_0 : i2, in %in_1 : i1, in %in_2 : i1, in %in_3 : i8, in %in_4 : i8, in %in_5 : i8, in %in_6 : i8, in %in_7 : i8, in %in_8 : i1, in %in_9 : i8, in %in_10 : i8, in %in_11 : i8, in %in_12 : i1, in %in_13 : i1, out out_0 : i1) attributes {"#pragma" = "CONTROL_NODE"} {
    %c115_i8 = hw.constant 115 : i8
    %c99_i8 = hw.constant 99 : i8
    %c51_i8 = hw.constant 51 : i8
    %c19_i8 = hw.constant 19 : i8
    %c1_i8 = hw.constant 1 : i8
    %c5_i8 = hw.constant 5 : i8
    %c0_i8 = hw.constant 0 : i8
    %true = hw.constant true
    %0 = comb.xor %in_1, %true : i1
    %1 = comb.icmp eq %in_4, %c0_i8 : i8
    %2 = comb.xor %1, %true : i1
    %3 = comb.icmp eq %in_7, %c0_i8 : i8
    %4 = comb.xor %3, %true : i1
    %5 = comb.icmp eq %in_6, %c5_i8 : i8
    %6 = comb.or %1, %5 : i1
    %7 = comb.xor %6, %true : i1
    %8 = comb.xor %in_8, %true : i1
    %9 = comb.icmp eq %in_10, %c1_i8 : i8
    %10 = comb.xor %9, %true : i1
    %11 = comb.icmp eq %in_3, %c19_i8 : i8
    %12 = comb.and %in_2, %11 : i1
    %13 = comb.and %12, %1 : i1
    %14 = comb.and %13, %in_1 : i1
    %15 = comb.and %13, %0 : i1
    %16 = comb.or %14, %15 : i1
    %17 = comb.and %12, %2 : i1
    %18 = comb.icmp eq %in_5, %c51_i8 : i8
    %19 = comb.and %in_2, %18 : i1
    %20 = comb.and %19, %1 : i1
    %21 = comb.and %20, %3 : i1
    %22 = comb.and %21, %in_1 : i1
    %23 = comb.and %21, %0 : i1
    %24 = comb.or %22, %23 : i1
    %25 = comb.and %20, %4 : i1
    %26 = comb.or %25, %24 : i1
    %27 = comb.and %19, %5 : i1
    %28 = comb.and %19, %7 : i1
    %29 = comb.icmp eq %in_9, %c99_i8 : i8
    %30 = comb.and %in_2, %29 : i1
    %31 = comb.and %30, %9 : i1
    %32 = comb.and %31, %in_8 : i1
    %33 = comb.and %31, %8 : i1
    %34 = comb.and %30, %10 : i1
    %35 = comb.icmp eq %in_11, %c115_i8 : i8
    %36 = comb.and %in_2, %35 : i1
    %37 = comb.or %11, %18, %29, %35 : i1
    %38 = comb.xor %37, %true : i1
    %39 = comb.and %in_2, %38 : i1
    %40 = comb.or %16, %17, %28, %26, %27, %32, %33, %34, %36, %39 : i1
    %41 = comb.xor %in_12, %true : i1
    %42 = comb.and %in_12, %in_13 : i1
    %43 = comb.xor %in_13, %true : i1
    %44 = comb.and %in_12, %43 : i1
    %45 = comb.or %42, %44, %41 : i1
    %46 = comb.and %40, %45 : i1
    %47 = comb.concat %46, %in_0 : i1, i2
    %LUT = SpecHLS.lookUpTable [%47 : i3] :i3= {0,0,0,0,1,2,3,4 }
    %48 = comb.concat %LUT, %16 : i3, i1
    %LUT_0 = SpecHLS.lookUpTable [%48 : i4] :i3= {0,0,1,1,2,3,4,4,5,5,5,5,5,5,5,5 }
    %49 = comb.concat %LUT_0, %14 : i3, i1
    %LUT_1 = SpecHLS.lookUpTable [%49 : i4] :i3= {0,0,1,1,2,2,3,4,5,5,6,6,6,6,6,6 }
    %50 = comb.concat %LUT_1, %26 : i3, i1
    %LUT_2 = SpecHLS.lookUpTable [%50 : i4] :i3= {0,0,1,1,2,2,3,3,4,4,5,6,7,7,7,7 }
    %51 = comb.concat %LUT_2, %24 : i3, i1
    %LUT_3 = SpecHLS.lookUpTable [%51 : i4] :i4= {0,0,1,1,2,2,3,3,4,4,5,5,6,7,8,8 }
    %52 = comb.concat %LUT_3, %22 : i4, i1
    %LUT_4 = SpecHLS.lookUpTable [%52 : i5] :i4= {0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9 }
    %LUT_5 = SpecHLS.lookUpTable [%LUT_4 : i4] :i2= {0,0,0,0,1,0,0,0,2,3,3,3,3,3,3,3 }
    %LUT_6 = SpecHLS.lookUpTable [%LUT_5 : i2] :i1= {0,0,1,0 }
    %LUT_7 = SpecHLS.lookUpTable [%LUT_6 : i1] :i1= {0,1 }
    hw.output %LUT_7 : i1
  }
  hw.module @SCC_0_ctrl_13(in %in_0 : i2, in %in_1 : i1, in %in_2 : i1, in %in_3 : i8, in %in_4 : i8, in %in_5 : i8, in %in_6 : i8, in %in_7 : i8, in %in_8 : i1, in %in_9 : i8, in %in_10 : i8, in %in_11 : i8, in %in_12 : i1, in %in_13 : i1, out out_0 : i1) attributes {"#pragma" = "CONTROL_NODE"} {
    %c115_i8 = hw.constant 115 : i8
    %c99_i8 = hw.constant 99 : i8
    %c51_i8 = hw.constant 51 : i8
    %c19_i8 = hw.constant 19 : i8
    %c1_i8 = hw.constant 1 : i8
    %c5_i8 = hw.constant 5 : i8
    %c0_i8 = hw.constant 0 : i8
    %true = hw.constant true
    %0 = comb.xor %in_1, %true : i1
    %1 = comb.icmp eq %in_4, %c0_i8 : i8
    %2 = comb.xor %1, %true : i1
    %3 = comb.icmp eq %in_7, %c0_i8 : i8
    %4 = comb.xor %3, %true : i1
    %5 = comb.icmp eq %in_6, %c5_i8 : i8
    %6 = comb.or %1, %5 : i1
    %7 = comb.xor %6, %true : i1
    %8 = comb.xor %in_8, %true : i1
    %9 = comb.icmp eq %in_10, %c1_i8 : i8
    %10 = comb.xor %9, %true : i1
    %11 = comb.icmp eq %in_3, %c19_i8 : i8
    %12 = comb.and %in_2, %11 : i1
    %13 = comb.and %12, %1 : i1
    %14 = comb.and %13, %in_1 : i1
    %15 = comb.and %13, %0 : i1
    %16 = comb.or %14, %15 : i1
    %17 = comb.and %12, %2 : i1
    %18 = comb.icmp eq %in_5, %c51_i8 : i8
    %19 = comb.and %in_2, %18 : i1
    %20 = comb.and %19, %1 : i1
    %21 = comb.and %20, %3 : i1
    %22 = comb.and %21, %in_1 : i1
    %23 = comb.and %21, %0 : i1
    %24 = comb.or %22, %23 : i1
    %25 = comb.and %20, %4 : i1
    %26 = comb.or %25, %24 : i1
    %27 = comb.and %19, %5 : i1
    %28 = comb.and %19, %7 : i1
    %29 = comb.icmp eq %in_9, %c99_i8 : i8
    %30 = comb.and %in_2, %29 : i1
    %31 = comb.and %30, %9 : i1
    %32 = comb.and %31, %in_8 : i1
    %33 = comb.and %31, %8 : i1
    %34 = comb.and %30, %10 : i1
    %35 = comb.icmp eq %in_11, %c115_i8 : i8
    %36 = comb.and %in_2, %35 : i1
    %37 = comb.or %11, %18, %29, %35 : i1
    %38 = comb.xor %37, %true : i1
    %39 = comb.and %in_2, %38 : i1
    %40 = comb.or %16, %17, %28, %26, %27, %32, %33, %34, %36, %39 : i1
    %41 = comb.xor %in_12, %true : i1
    %42 = comb.and %in_12, %in_13 : i1
    %43 = comb.xor %in_13, %true : i1
    %44 = comb.and %in_12, %43 : i1
    %45 = comb.or %42, %44, %41 : i1
    %46 = comb.and %40, %45 : i1
    %47 = comb.concat %46, %in_0 : i1, i2
    %LUT = SpecHLS.lookUpTable [%47 : i3] :i3= {0,0,0,0,1,2,3,4 }
    %48 = comb.concat %LUT, %16 : i3, i1
    %LUT_0 = SpecHLS.lookUpTable [%48 : i4] :i3= {0,0,1,1,2,3,4,4,5,5,5,5,5,5,5,5 }
    %49 = comb.concat %LUT_0, %14 : i3, i1
    %LUT_1 = SpecHLS.lookUpTable [%49 : i4] :i3= {0,0,1,1,2,2,3,4,5,5,6,6,6,6,6,6 }
    %50 = comb.concat %LUT_1, %26 : i3, i1
    %LUT_2 = SpecHLS.lookUpTable [%50 : i4] :i3= {0,0,1,1,2,2,3,3,4,4,5,6,7,7,7,7 }
    %51 = comb.concat %LUT_2, %24 : i3, i1
    %LUT_3 = SpecHLS.lookUpTable [%51 : i4] :i4= {0,0,1,1,2,2,3,3,4,4,5,5,6,7,8,8 }
    %52 = comb.concat %LUT_3, %22 : i4, i1
    %LUT_4 = SpecHLS.lookUpTable [%52 : i5] :i4= {0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9 }
    %LUT_5 = SpecHLS.lookUpTable [%LUT_4 : i4] :i2= {0,0,0,0,1,0,0,0,2,3,3,3,3,3,3,3 }
    %LUT_6 = SpecHLS.lookUpTable [%LUT_5 : i2] :i1= {0,0,1,0 }
    hw.output %LUT_6 : i1
  }
  hw.module @SCC_0_ctrl_14(in %in_0 : i2, in %in_1 : i1, in %in_2 : i1, in %in_3 : i8, in %in_4 : i8, in %in_5 : i8, in %in_6 : i8, in %in_7 : i8, in %in_8 : i1, in %in_9 : i8, in %in_10 : i8, in %in_11 : i8, in %in_12 : i1, in %in_13 : i1, out out_0 : i1) attributes {"#pragma" = "CONTROL_NODE"} {
    %c115_i8 = hw.constant 115 : i8
    %c99_i8 = hw.constant 99 : i8
    %c51_i8 = hw.constant 51 : i8
    %c19_i8 = hw.constant 19 : i8
    %c1_i8 = hw.constant 1 : i8
    %c5_i8 = hw.constant 5 : i8
    %c0_i8 = hw.constant 0 : i8
    %true = hw.constant true
    %0 = comb.xor %in_1, %true : i1
    %1 = comb.icmp eq %in_4, %c0_i8 : i8
    %2 = comb.xor %1, %true : i1
    %3 = comb.icmp eq %in_7, %c0_i8 : i8
    %4 = comb.xor %3, %true : i1
    %5 = comb.icmp eq %in_6, %c5_i8 : i8
    %6 = comb.or %1, %5 : i1
    %7 = comb.xor %6, %true : i1
    %8 = comb.xor %in_8, %true : i1
    %9 = comb.icmp eq %in_10, %c1_i8 : i8
    %10 = comb.xor %9, %true : i1
    %11 = comb.icmp eq %in_3, %c19_i8 : i8
    %12 = comb.and %in_2, %11 : i1
    %13 = comb.and %12, %1 : i1
    %14 = comb.and %13, %in_1 : i1
    %15 = comb.and %13, %0 : i1
    %16 = comb.or %14, %15 : i1
    %17 = comb.and %12, %2 : i1
    %18 = comb.icmp eq %in_5, %c51_i8 : i8
    %19 = comb.and %in_2, %18 : i1
    %20 = comb.and %19, %1 : i1
    %21 = comb.and %20, %3 : i1
    %22 = comb.and %21, %in_1 : i1
    %23 = comb.and %21, %0 : i1
    %24 = comb.or %22, %23 : i1
    %25 = comb.and %20, %4 : i1
    %26 = comb.or %25, %24 : i1
    %27 = comb.and %19, %5 : i1
    %28 = comb.and %19, %7 : i1
    %29 = comb.icmp eq %in_9, %c99_i8 : i8
    %30 = comb.and %in_2, %29 : i1
    %31 = comb.and %30, %9 : i1
    %32 = comb.and %31, %in_8 : i1
    %33 = comb.and %31, %8 : i1
    %34 = comb.and %30, %10 : i1
    %35 = comb.icmp eq %in_11, %c115_i8 : i8
    %36 = comb.and %in_2, %35 : i1
    %37 = comb.or %11, %18, %29, %35 : i1
    %38 = comb.xor %37, %true : i1
    %39 = comb.and %in_2, %38 : i1
    %40 = comb.or %16, %17, %28, %26, %27, %32, %33, %34, %36, %39 : i1
    %41 = comb.xor %in_12, %true : i1
    %42 = comb.and %in_12, %in_13 : i1
    %43 = comb.xor %in_13, %true : i1
    %44 = comb.and %in_12, %43 : i1
    %45 = comb.or %42, %44, %41 : i1
    %46 = comb.and %40, %45 : i1
    %47 = comb.concat %46, %in_0 : i1, i2
    %LUT = SpecHLS.lookUpTable [%47 : i3] :i3= {0,0,0,0,1,2,3,4 }
    %48 = comb.concat %LUT, %16 : i3, i1
    %LUT_0 = SpecHLS.lookUpTable [%48 : i4] :i3= {0,0,1,1,2,3,4,4,5,5,5,5,5,5,5,5 }
    %49 = comb.concat %LUT_0, %14 : i3, i1
    %LUT_1 = SpecHLS.lookUpTable [%49 : i4] :i3= {0,0,1,1,2,2,3,4,5,5,6,6,6,6,6,6 }
    %50 = comb.concat %LUT_1, %26 : i3, i1
    %LUT_2 = SpecHLS.lookUpTable [%50 : i4] :i3= {0,0,1,1,2,2,3,3,4,4,5,6,7,7,7,7 }
    %51 = comb.concat %LUT_2, %24 : i3, i1
    %LUT_3 = SpecHLS.lookUpTable [%51 : i4] :i4= {0,0,1,1,2,2,3,3,4,4,5,5,6,7,8,8 }
    %52 = comb.concat %LUT_3, %22 : i4, i1
    %LUT_4 = SpecHLS.lookUpTable [%52 : i5] :i4= {0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9 }
    %LUT_5 = SpecHLS.lookUpTable [%LUT_4 : i4] :i2= {0,0,0,0,1,0,0,0,2,3,3,3,3,3,3,3 }
    %LUT_6 = SpecHLS.lookUpTable [%LUT_5 : i2] :i1= {0,0,1,0 }
    hw.output %LUT_6 : i1
  }
  hw.module @SCC_0_ctrl_15(in %in_0 : i2, in %in_1 : i1, in %in_2 : i1, in %in_3 : i8, in %in_4 : i8, in %in_5 : i8, in %in_6 : i8, in %in_7 : i8, in %in_8 : i1, in %in_9 : i8, in %in_10 : i8, in %in_11 : i8, in %in_12 : i1, in %in_13 : i1, out out_0 : i1) attributes {"#pragma" = "CONTROL_NODE"} {
    %c115_i8 = hw.constant 115 : i8
    %c99_i8 = hw.constant 99 : i8
    %c51_i8 = hw.constant 51 : i8
    %c19_i8 = hw.constant 19 : i8
    %c1_i8 = hw.constant 1 : i8
    %c5_i8 = hw.constant 5 : i8
    %c0_i8 = hw.constant 0 : i8
    %true = hw.constant true
    %0 = comb.xor %in_1, %true : i1
    %1 = comb.icmp eq %in_4, %c0_i8 : i8
    %2 = comb.xor %1, %true : i1
    %3 = comb.icmp eq %in_7, %c0_i8 : i8
    %4 = comb.xor %3, %true : i1
    %5 = comb.icmp eq %in_6, %c5_i8 : i8
    %6 = comb.or %1, %5 : i1
    %7 = comb.xor %6, %true : i1
    %8 = comb.xor %in_8, %true : i1
    %9 = comb.icmp eq %in_10, %c1_i8 : i8
    %10 = comb.xor %9, %true : i1
    %11 = comb.icmp eq %in_3, %c19_i8 : i8
    %12 = comb.and %in_2, %11 : i1
    %13 = comb.and %12, %1 : i1
    %14 = comb.and %13, %in_1 : i1
    %15 = comb.and %13, %0 : i1
    %16 = comb.or %14, %15 : i1
    %17 = comb.and %12, %2 : i1
    %18 = comb.icmp eq %in_5, %c51_i8 : i8
    %19 = comb.and %in_2, %18 : i1
    %20 = comb.and %19, %1 : i1
    %21 = comb.and %20, %3 : i1
    %22 = comb.and %21, %in_1 : i1
    %23 = comb.and %21, %0 : i1
    %24 = comb.or %22, %23 : i1
    %25 = comb.and %20, %4 : i1
    %26 = comb.or %25, %24 : i1
    %27 = comb.and %19, %5 : i1
    %28 = comb.and %19, %7 : i1
    %29 = comb.icmp eq %in_9, %c99_i8 : i8
    %30 = comb.and %in_2, %29 : i1
    %31 = comb.and %30, %9 : i1
    %32 = comb.and %31, %in_8 : i1
    %33 = comb.and %31, %8 : i1
    %34 = comb.and %30, %10 : i1
    %35 = comb.icmp eq %in_11, %c115_i8 : i8
    %36 = comb.and %in_2, %35 : i1
    %37 = comb.or %11, %18, %29, %35 : i1
    %38 = comb.xor %37, %true : i1
    %39 = comb.and %in_2, %38 : i1
    %40 = comb.or %16, %17, %28, %26, %27, %32, %33, %34, %36, %39 : i1
    %41 = comb.xor %in_12, %true : i1
    %42 = comb.and %in_12, %in_13 : i1
    %43 = comb.xor %in_13, %true : i1
    %44 = comb.and %in_12, %43 : i1
    %45 = comb.or %42, %44, %41 : i1
    %46 = comb.and %40, %45 : i1
    %47 = comb.concat %46, %in_0 : i1, i2
    %LUT = SpecHLS.lookUpTable [%47 : i3] :i3= {0,0,0,0,1,2,3,4 }
    %48 = comb.concat %LUT, %16 : i3, i1
    %LUT_0 = SpecHLS.lookUpTable [%48 : i4] :i3= {0,0,1,1,2,3,4,4,5,5,5,5,5,5,5,5 }
    %49 = comb.concat %LUT_0, %14 : i3, i1
    %LUT_1 = SpecHLS.lookUpTable [%49 : i4] :i3= {0,0,1,1,2,2,3,4,5,5,6,6,6,6,6,6 }
    %50 = comb.concat %LUT_1, %26 : i3, i1
    %LUT_2 = SpecHLS.lookUpTable [%50 : i4] :i3= {0,0,1,1,2,2,3,3,4,4,5,6,7,7,7,7 }
    %51 = comb.concat %LUT_2, %24 : i3, i1
    %LUT_3 = SpecHLS.lookUpTable [%51 : i4] :i4= {0,0,1,1,2,2,3,3,4,4,5,5,6,7,8,8 }
    %52 = comb.concat %LUT_3, %22 : i4, i1
    %LUT_4 = SpecHLS.lookUpTable [%52 : i5] :i4= {0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9 }
    %LUT_5 = SpecHLS.lookUpTable [%LUT_4 : i4] :i2= {0,0,0,0,1,0,0,0,2,3,3,3,3,3,3,3 }
    %LUT_6 = SpecHLS.lookUpTable [%LUT_5 : i2] :i1= {0,0,1,0 }
    hw.output %LUT_6 : i1
  }
  hw.module @SCC_0_ctrl_16(in %in_0 : i2, in %in_1 : i1, in %in_2 : i1, in %in_3 : i8, in %in_4 : i8, in %in_5 : i8, in %in_6 : i8, in %in_7 : i8, in %in_8 : i1, in %in_9 : i8, in %in_10 : i8, in %in_11 : i8, in %in_12 : i1, in %in_13 : i1, out out_0 : i1) attributes {"#pragma" = "CONTROL_NODE"} {
    %c115_i8 = hw.constant 115 : i8
    %c99_i8 = hw.constant 99 : i8
    %c51_i8 = hw.constant 51 : i8
    %c19_i8 = hw.constant 19 : i8
    %c1_i8 = hw.constant 1 : i8
    %c5_i8 = hw.constant 5 : i8
    %c0_i8 = hw.constant 0 : i8
    %true = hw.constant true
    %0 = comb.xor %in_1, %true : i1
    %1 = comb.icmp eq %in_4, %c0_i8 : i8
    %2 = comb.xor %1, %true : i1
    %3 = comb.icmp eq %in_7, %c0_i8 : i8
    %4 = comb.xor %3, %true : i1
    %5 = comb.icmp eq %in_6, %c5_i8 : i8
    %6 = comb.or %1, %5 : i1
    %7 = comb.xor %6, %true : i1
    %8 = comb.xor %in_8, %true : i1
    %9 = comb.icmp eq %in_10, %c1_i8 : i8
    %10 = comb.xor %9, %true : i1
    %11 = comb.icmp eq %in_3, %c19_i8 : i8
    %12 = comb.and %in_2, %11 : i1
    %13 = comb.and %12, %1 : i1
    %14 = comb.and %13, %in_1 : i1
    %15 = comb.and %13, %0 : i1
    %16 = comb.or %14, %15 : i1
    %17 = comb.and %12, %2 : i1
    %18 = comb.icmp eq %in_5, %c51_i8 : i8
    %19 = comb.and %in_2, %18 : i1
    %20 = comb.and %19, %1 : i1
    %21 = comb.and %20, %3 : i1
    %22 = comb.and %21, %in_1 : i1
    %23 = comb.and %21, %0 : i1
    %24 = comb.or %22, %23 : i1
    %25 = comb.and %20, %4 : i1
    %26 = comb.or %25, %24 : i1
    %27 = comb.and %19, %5 : i1
    %28 = comb.and %19, %7 : i1
    %29 = comb.icmp eq %in_9, %c99_i8 : i8
    %30 = comb.and %in_2, %29 : i1
    %31 = comb.and %30, %9 : i1
    %32 = comb.and %31, %in_8 : i1
    %33 = comb.and %31, %8 : i1
    %34 = comb.and %30, %10 : i1
    %35 = comb.icmp eq %in_11, %c115_i8 : i8
    %36 = comb.and %in_2, %35 : i1
    %37 = comb.or %11, %18, %29, %35 : i1
    %38 = comb.xor %37, %true : i1
    %39 = comb.and %in_2, %38 : i1
    %40 = comb.or %16, %17, %28, %26, %27, %32, %33, %34, %36, %39 : i1
    %41 = comb.xor %in_12, %true : i1
    %42 = comb.and %in_12, %in_13 : i1
    %43 = comb.xor %in_13, %true : i1
    %44 = comb.and %in_12, %43 : i1
    %45 = comb.or %42, %44, %41 : i1
    %46 = comb.and %40, %45 : i1
    %47 = comb.concat %46, %in_0 : i1, i2
    %LUT = SpecHLS.lookUpTable [%47 : i3] :i3= {0,0,0,0,1,2,3,4 }
    %48 = comb.concat %LUT, %16 : i3, i1
    %LUT_0 = SpecHLS.lookUpTable [%48 : i4] :i3= {0,0,1,1,2,3,4,4,5,5,5,5,5,5,5,5 }
    %49 = comb.concat %LUT_0, %14 : i3, i1
    %LUT_1 = SpecHLS.lookUpTable [%49 : i4] :i3= {0,0,1,1,2,2,3,4,5,5,6,6,6,6,6,6 }
    %50 = comb.concat %LUT_1, %26 : i3, i1
    %LUT_2 = SpecHLS.lookUpTable [%50 : i4] :i3= {0,0,1,1,2,2,3,3,4,4,5,6,7,7,7,7 }
    %51 = comb.concat %LUT_2, %24 : i3, i1
    %LUT_3 = SpecHLS.lookUpTable [%51 : i4] :i4= {0,0,1,1,2,2,3,3,4,4,5,5,6,7,8,8 }
    %52 = comb.concat %LUT_3, %22 : i4, i1
    %LUT_4 = SpecHLS.lookUpTable [%52 : i5] :i4= {0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9 }
    %LUT_5 = SpecHLS.lookUpTable [%LUT_4 : i4] :i2= {0,0,0,0,1,0,0,0,2,3,3,3,3,3,3,3 }
    %LUT_6 = SpecHLS.lookUpTable [%LUT_5 : i2] :i1= {0,0,1,0 }
    hw.output %LUT_6 : i1
  }
  hw.module @SCC_0_ctrl_17(in %in_0 : i2, in %in_1 : i1, in %in_2 : i1, in %in_3 : i8, in %in_4 : i8, in %in_5 : i8, in %in_6 : i8, in %in_7 : i8, in %in_8 : i1, in %in_9 : i8, in %in_10 : i8, in %in_11 : i8, in %in_12 : i1, in %in_13 : i1, out out_0 : i2) attributes {"#pragma" = "CONTROL_NODE"} {
    %c115_i8 = hw.constant 115 : i8
    %c99_i8 = hw.constant 99 : i8
    %c51_i8 = hw.constant 51 : i8
    %c19_i8 = hw.constant 19 : i8
    %c1_i8 = hw.constant 1 : i8
    %c5_i8 = hw.constant 5 : i8
    %c0_i8 = hw.constant 0 : i8
    %true = hw.constant true
    %0 = comb.xor %in_1, %true : i1
    %1 = comb.icmp eq %in_4, %c0_i8 : i8
    %2 = comb.xor %1, %true : i1
    %3 = comb.icmp eq %in_7, %c0_i8 : i8
    %4 = comb.xor %3, %true : i1
    %5 = comb.icmp eq %in_6, %c5_i8 : i8
    %6 = comb.or %1, %5 : i1
    %7 = comb.xor %6, %true : i1
    %8 = comb.xor %in_8, %true : i1
    %9 = comb.icmp eq %in_10, %c1_i8 : i8
    %10 = comb.xor %9, %true : i1
    %11 = comb.icmp eq %in_3, %c19_i8 : i8
    %12 = comb.and %in_2, %11 : i1
    %13 = comb.and %12, %1 : i1
    %14 = comb.and %13, %in_1 : i1
    %15 = comb.and %13, %0 : i1
    %16 = comb.or %14, %15 : i1
    %17 = comb.and %12, %2 : i1
    %18 = comb.icmp eq %in_5, %c51_i8 : i8
    %19 = comb.and %in_2, %18 : i1
    %20 = comb.and %19, %1 : i1
    %21 = comb.and %20, %3 : i1
    %22 = comb.and %21, %in_1 : i1
    %23 = comb.and %21, %0 : i1
    %24 = comb.or %22, %23 : i1
    %25 = comb.and %20, %4 : i1
    %26 = comb.or %25, %24 : i1
    %27 = comb.and %19, %5 : i1
    %28 = comb.and %19, %7 : i1
    %29 = comb.icmp eq %in_9, %c99_i8 : i8
    %30 = comb.and %in_2, %29 : i1
    %31 = comb.and %30, %9 : i1
    %32 = comb.and %31, %in_8 : i1
    %33 = comb.and %31, %8 : i1
    %34 = comb.and %30, %10 : i1
    %35 = comb.icmp eq %in_11, %c115_i8 : i8
    %36 = comb.and %in_2, %35 : i1
    %37 = comb.or %11, %18, %29, %35 : i1
    %38 = comb.xor %37, %true : i1
    %39 = comb.and %in_2, %38 : i1
    %40 = comb.or %16, %17, %28, %26, %27, %32, %33, %34, %36, %39 : i1
    %41 = comb.xor %in_12, %true : i1
    %42 = comb.and %in_12, %in_13 : i1
    %43 = comb.xor %in_13, %true : i1
    %44 = comb.and %in_12, %43 : i1
    %45 = comb.or %42, %44, %41 : i1
    %46 = comb.and %40, %45 : i1
    %47 = comb.concat %46, %in_0 : i1, i2
    %LUT = SpecHLS.lookUpTable [%47 : i3] :i3= {0,0,0,0,1,2,3,4 }
    %48 = comb.concat %LUT, %16 : i3, i1
    %LUT_0 = SpecHLS.lookUpTable [%48 : i4] :i3= {0,0,1,1,2,3,4,4,5,5,5,5,5,5,5,5 }
    %49 = comb.concat %LUT_0, %14 : i3, i1
    %LUT_1 = SpecHLS.lookUpTable [%49 : i4] :i3= {0,0,1,1,2,2,3,4,5,5,6,6,6,6,6,6 }
    %50 = comb.concat %LUT_1, %26 : i3, i1
    %LUT_2 = SpecHLS.lookUpTable [%50 : i4] :i3= {0,0,1,1,2,2,3,3,4,4,5,6,7,7,7,7 }
    %51 = comb.concat %LUT_2, %24 : i3, i1
    %LUT_3 = SpecHLS.lookUpTable [%51 : i4] :i4= {0,0,1,1,2,2,3,3,4,4,5,5,6,7,8,8 }
    %52 = comb.concat %LUT_3, %22 : i4, i1
    %LUT_4 = SpecHLS.lookUpTable [%52 : i5] :i4= {0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9 }
    %LUT_5 = SpecHLS.lookUpTable [%LUT_4 : i4] :i2= {0,0,0,0,1,0,0,0,2,3,3,3,3,3,3,3 }
    hw.output %LUT_5 : i2
  }

}
