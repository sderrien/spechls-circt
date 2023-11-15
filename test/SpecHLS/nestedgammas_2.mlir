// RUN: spechls-opt --merge-gammas %s | spechls-opt | FileCheck %s
// CHECK-LABEL:   @bar
module {
  func.func @bar(%arg0: i2, %arg1: i2, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) -> i32 {
    %0 = comb.concat %arg1, %arg0 : i2, i2
    %1 = SpecHLS.lookUpTable [%0 ] :i3= {0,1,2,3,4,4,4,4,5,5,5,5,6,6,6,6 }
    %2 = SpecHLS.gamma %1 ? %arg2,%arg3,%arg4,%arg2,%arg4 :i32
    return %2 : i32
  }
}
