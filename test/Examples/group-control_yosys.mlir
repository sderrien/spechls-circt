// RUN: spechls-opt --group-control --canonicalize %s | spechls-opt --yosys-optimizer="replace-with-optimized-module=true" | FileCheck %s
// CHECK-LABEL:   @SCC_0ctrl_1
// CHECK-SAME:    (%[[ARG0:.*]]: i1, %[[ARG1:.*]]: i1, %[[ARG2:.*]]: i1, %[[ARG3:.*]]: i1) -> i1 {
// CHECK-NEXT:         %0 = comb.and %[[ARG2:.*]], %[[ARG3:.*]] : i1
// CHECK-NEXT:         %1 = comb.or %[[ARG0:.*]], %[[ARG1:.*]] : i1
// CHECK-NEXT:         %2 = comb.xor %0, %1 : i1
// CHECK-NEXT:         hw.output %2 : i1
// module {
// module {
//     hw.module @SCC_0ctrl_1(in %in_0 : i1, in %in_1 : i1, in %in_2 : i1, in %in_3 : i1, out out_0 : i1) attributes {"#pragma" = "CONTROL_NODE"} {
//       %0 = comb.and %in_2, %in_3 : i1
//       %1 = comb.or %in_0, %in_1 : i1
//       %2 = comb.xor %0, %1 : i1
//       hw.output %2 : i1
//     }
//     hw.module @SCC_0(out out_0 : i1) {
//       %0 = SpecHLS.init "a" : i1
//       %1 = SpecHLS.init "b" : i1
//       %2 = SpecHLS.init "c" : i1
//       %3 = SpecHLS.init "d" : i1
//       %4 = SpecHLS.init "e" : i1
//       %5 = SpecHLS.init "f" : i1
//       %SCC_0ctrl_1.out_0 = hw.instance "SCC_0ctrl_1" @SCC_0ctrl_1(in_0: %2: i1, in_1: %3: i1, in_2: %0: i1, in_3: %1: i1) -> (out_0: i1)
//       %6 = SpecHLS.gamma"dummy" %SCC_0ctrl_1.out_0 ?  {name = "dummy"}%4,%5 :i1
//       %7 = SpecHLS.exit %6
//       hw.output %7 : i1
//     }
//     hw.module @SCC_1ctrl_2(in %in_0 : i1, in %in_1 : i1, in %in_2 : i1, in %in_3 : i1, out out_0 : i1) attributes {"#pragma" = "CONTROL_NODE"} {
//       %0 = comb.and %in_2, %in_3 : i1
//       %1 = comb.or %in_0, %in_1 : i1
//       %2 = comb.xor %0, %1 : i1
//       hw.output %2 : i1
//     }
//     hw.module @SCC_1(in %0 "" : i1, in %1 "" : i1, in %2 "" : i1, in %3 "" : i1, in %4 "" : i1, in %5 "" : i1, out out_0 : i1) {
//       %SCC_1ctrl_2.out_0 = hw.instance "SCC_1ctrl_2" @SCC_1ctrl_2(in_0: %2: i1, in_1: %3: i1, in_2: %0: i1, in_3: %1: i1) -> (out_0: i1)
//       %6 = SpecHLS.gamma"undef" %SCC_1ctrl_2.out_0 ?  {name = "undef"}%4,%5 :i1
//       hw.output %6 : i1
//     }
//   }
//
//

module {
  hw.module @SCC_0(out out_0 : i1) {
    %0 = SpecHLS.init @a : i1
    %1 = SpecHLS.init @b : i1
    %2 = SpecHLS.init @c : i1
    %3 = SpecHLS.init @d : i1

    %6 = comb.and %0, %1 : i1
    %7 = comb.or %2, %3 : i1
    %8 = comb.xor %1, %2 : i1

    %11 = SpecHLS.gamma @dummy %8 ? %4,%5 :i1
    %51 = SpecHLS.exit %11
    hw.output %51 :i1
  }

  hw.module @SCC_1(in %0 : i1, in %1 : i1, in %2 : i1, in %3 : i1, in %4 : i1, in %5 : i1, out out_0 : i1) {
    %6 = comb.and %0, %1 : i1
    %7 = comb.or %2, %3 : i1
    %8 = comb.xor %6, %7 : i1
    %11 = SpecHLS.gamma @x %8 ? %4,%5 :i1
    hw.output %11 :i1
  }

}

