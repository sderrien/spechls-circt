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
  }