// RUN: spechls-opt --yosys-optimizer="replace-with-optimized-module=true"  %s | spechls-opt | FileCheck %s
module {

  hw.module @SCC_0ctrl_1(in %in_0 : i1, in %in_1 : i1, in %in_2 : i1, in %in_3 : i1, out out_0 : i1) attributes {"#pragma" = "CONTROL_NODE"} {
    %0 = comb.and %in_0, %in_1 : i1
    %1 = comb.or %in_2, %in_3 : i1
    %2 = comb.xor %0, %1 : i1
    %3 = comb.concat %0, %1, %2 : i1, i1, i1
    %4 = SpecHLS.lookUpTable [%3 ] :i1= {0,1,1,0,1,0,1,0 }
    hw.output %4 : i1
  }

  hw.module @SCC_0(out out_0 : i1) {
    %0 = SpecHLS.init @a : i1
    %1 = SpecHLS.init @b : i1
    %2 = SpecHLS.init @c : i1
    %3 = SpecHLS.init @d : i1
    %4 = SpecHLS.init @e : i1
    %5 = SpecHLS.init @f : i1
    %SCC_0ctrl_1.out_0 = hw.instance "SCC_0ctrl_1" @SCC_0ctrl_1(in_0: %0: i1, in_1: %1: i1, in_2: %2: i1, in_3: %3: i1) -> (out_0: i1)
    %6 = SpecHLS.gamma @dummy %SCC_0ctrl_1.out_0 ?  %4,%5 :i1
    %7 = SpecHLS.exit %6
    hw.output %7 : i1
  }
  hw.module @SCC_1ctrl_2(in %in_0 : i1, in %in_1 : i1, in %in_2 : i1, in %in_3 : i1, out out_0 : i1) attributes {"#pragma" = "CONTROL_NODE"} {
    %0 = comb.and %in_2, %in_3 : i1
    %1 = comb.or %in_0, %in_1 : i1
    %2 = comb.xor %0, %1 : i1
    hw.output %2 : i1
  }
  hw.module @SCC_1(in %0 "" : i1, in %1 "" : i1, in %2 "" : i1, in %3 "" : i1, in %4 "" : i1, in %5 "" : i1, out out_0 : i1) {
    %SCC_1ctrl_2.out_0 = hw.instance "SCC_1ctrl_2" @SCC_1ctrl_2(in_0: %2: i1, in_1: %3: i1, in_2: %0: i1, in_3: %1: i1) -> (out_0: i1)
    %6 = SpecHLS.gamma @x %SCC_1ctrl_2.out_0 ? %4,%5 :i1
    hw.output %6 : i1
  }
}
