module {
hw.module @SCC_0() {
  %0 = SpecHLS.init @x0 : i1
  %1 = SpecHLS.init @x1 : i1
  %2 = SpecHLS.init @x2 : i1
  %3 = SpecHLS.init @x3 : i1
  %4 = SpecHLS.init @x4 : i1
  %5 = SpecHLS.init @c0 : i1
  %6 = SpecHLS.init @c1 : i2
  %8 = SpecHLS.gamma @i %5 ? %1,%0 :i1

  %10 = SpecHLS.init @c3 : i2
  %11 = SpecHLS.gamma @i %10 ? %4,%8,%5 :i1
  %12 = SpecHLS.exit %11
  hw.output
}

hw.module @SCC_1(
  in %c0 : i2, in %c1 : i1,
  in %x0 : i32, in %x1 : i32, in %x2 : i32, in %x3 : i32, in %x4 : i32, 
  out result : i32)
{
  %0 = SpecHLS.gamma @g0 %c0 ? %x0, %x1, %x2 : i32
  %1 = SpecHLS.gamma @g1 %c0 ? %x1, %x2, %x3, %x4 : i32
  %result = SpecHLS.gamma @g2 %c1 ? %0, %1 : i32
  hw.output %result : i32
}

}
