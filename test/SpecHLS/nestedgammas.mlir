// RUN: spechls-opt --merge-gammas %s | spechls-opt | FileCheck %s
// CHECK-LABEL:   @bar
module {
func.func @bar(%a: i2,%b: i32,%c: i32,%d: i32, %e:i32) -> i32  {
        %res = SpecHLS.gamma %a ? %b, %c, %d :i32
        %res2 = SpecHLS.gamma %a ? %res, %c, %e :i32
        return %res2 : i32
}
}
