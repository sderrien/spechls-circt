// RUN: ./cmake-build-debug/bin/spechls-opt --group-control  ./test/SpecHLS/test-group-control.mlir | ./cmake-build-debug/bin/spechls-opt --yosys-optimizer="replace-with-optimized-module=true" | FileCheck %s
module {
  hw.module @SCC_0(out out_0 : i1) {
    %0 = SpecHLS.init @a : i1
    %1 = SpecHLS.init @b : i1
    %2 = SpecHLS.init @c : i1
    %3 = SpecHLS.init @d : i1
    %4 = SpecHLS.init @e : i1
    %5 = SpecHLS.init @f : i1

    %6 = comb.and %0, %1 : i1
    %7 = comb.or %2, %3 : i1
    %8 = comb.xor %6, %7 : i1
    %10 = comb.concat %6, %7, %8 : i1, i1,i1
    %11 = SpecHLS.lookUpTable [%10]:i1 = {0,1,1,0,1,0,1,0}
    %12 = SpecHLS.gamma @dummy %11 ? %4,%5 :i1
    %51 = SpecHLS.exit %12

    hw.output %51 :i1
  }

  hw.module @SCC_1(in %0 : i1, in %1 : i1, in %2 : i1, in %3 : i1, in %4 : i1, in %5 : i1, out out_0 : i1) {
    %6 = comb.and %0, %1 : i1
    %7 = comb.or %2, %3 : i1
    %8 = comb.xor %6, %7 : i1
    %11 = SpecHLS.gamma @x %8 ? %4,%5 :i1
    hw.output %11 :i1
  }

  hw.module @SCC_2(in %0 : i1, in %1 : i1, in %2 : i1, in %3 : i1, in %4 : i1, in %5 : i1, out out_0 : i1) {
    %6 = comb.and %0, %1 : i1
    %7 = comb.or %2, %3 : i1
    %8 = comb.xor %6, %6 : i1
    %11 = SpecHLS.gamma @x %8 ? %4,%5 :i1
    %12 = SpecHLS.exit %11
    hw.output %12 :i1
  }

}

