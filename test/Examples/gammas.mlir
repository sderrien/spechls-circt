// RUN: spechls-opt --lower-spechls-to-comb %s | spechls-opt | FileCheck %s
// CHECK-LABEL:   @bar
// CHECK-SAME:    (%[[ARG0:.*]]: i2, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[ARG2:.*]]: i32) -> i32 {
// CHECK-NEXT:         %0 = comb.extract %[[ARG0:.*]] from 0 : (i2) -> i1
// CHECK-NEXT:         %1 = comb.mux %0, %[[ARG1:.*]], %[[ARG2:.*]] : i32
// CHECK-NEXT:         %2 = comb.mux %0, %[[ARG3:.*]], %1 : i32
// CHECK-NEXT:         %3 = comb.extract %[[ARG0:.*]] from 1 : (i2) -> i1
// CHECK:         %4 = comb.mux %3, %1, %2 : i32
// CHECK:         return %4 : i32

module {
func.func @bar(%a: i2,%b: i32,%c: i32,%d: i32) -> i32  {
        %res = SpecHLS.gamma @x %a ? %b, %c, %d :i32
        return %res : i32
}
}
