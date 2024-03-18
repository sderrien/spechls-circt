// RUN: spechls-opt %s | spechls-opt | FileCheck %s
// CHECK:module {
// CHECK:}

module {
hw.module @bar(in %io : i32,in %we : i1,in %a : i32,in %b : i32, out out0 :i32) {
        %res1 = SpecHLS.ioprintf "%d" (%a: i32) from %io when %we
        hw.output %res1 : i32
}
}
