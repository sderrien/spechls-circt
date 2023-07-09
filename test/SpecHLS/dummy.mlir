// RUN: spechls-opt %s | spechls-opt | FileCheck %s
module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i1
        %1 = arith.constant 2 : i32
        %2 = arith.constant 3 : i32
        %3 = arith.constant 4 : i32
        %4 = arith.constant 5 : i32

        %res2 = SpecHLS.delay ( %1 : i32, %0 ,%2: i32) : i32
        %res = SpecHLS.gamma (%0 : i1, %1,%2: i32, i32) : i32
        return
    }
}
