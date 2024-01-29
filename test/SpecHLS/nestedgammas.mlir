// RUN: spechls-opt --merge-gammas %s | spechls-opt | FileCheck %s
// CHECK-LABEL:   @bar
// CHECK-SAME: (%arg0: i2, %arg1: i2, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) -> i32 {
// CHECK-NEXT:     %0 = comb.concat %arg1, %arg0 : i2, i2
// CHECK:     %1 = SpecHLS.lookUpTable [%0 ] :i3= {0,1,2,3,4,4,4,4,5,5,5,5,6,6,6,6 }
// CHECK:     %2 = SpecHLS.gamma %1 ? %arg2,%arg3,%arg4,%arg2,%arg4 :i32
// CHECK:    return %2 : i32

module {
func.func @bar(%a: i2,%b: i2 ,%c: i32,%d: i32, %e:i32,%f:i32) -> i32  {
        %res = SpecHLS.gamma @x %a ? %c, %d, %e :i32
        %res2 = SpecHLS.gamma @y %b ? %res, %c, %e :i32
        return %res2 : i32
}
}
