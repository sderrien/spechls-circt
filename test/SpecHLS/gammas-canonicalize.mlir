// RUN: spechls-opt --canonicalize %s | spechls-opt | FileCheck %s
// CHECK-LABEL:   @bar
// CHECK-SAME:    (%[[ARG0:.*]]: i2, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[ARG2:.*]]: i32) -> i32 {
// CHECK:         return %[[ARG1:.*]] : i32

module {
    func.func @bar(%a: i32,%b: i32,%c: i32,%d: i32) -> i32  {
            %0 = hw.constant 0x1 :i2
            %res = SpecHLS.gamma @x  %0 ? %a , %b, %c, %d :i32
            return %res : i32
    }
}
