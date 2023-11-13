// RUN: spechls-opt --lower-spechls-to-comb %s | spechls-opt | FileCheck %s
// CHECK-LABEL: @bar
// CHECK:    (%[[ARG0:.*]]: i3) -> i32 {
// CHECK-NEXT:          %[[RES:.*]] = SpecHLS.lookUpTable [%[[ARG0:.*]] ] :i32= {0,3,5,6,8,3334,4564,45 }
// CHECK:             return %[[RES:.*]] : i32
// CHECK:             }
// CHECK:             }



module {
func.func @bar(%a: i3) -> i32 {
        %res = SpecHLS.lookUpTable [%a]:i32 = {0,3,5,6,8,3334,4564,45}
        return %res : i32
}
}
