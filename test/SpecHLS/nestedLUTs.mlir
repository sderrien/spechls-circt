// RUN: spechls-opt --merge-luts %s | spechls-opt | FileCheck %s
// CHECK-LABEL: @bar
// CHECK:    (%[[ARG0:.*]]: i3) -> i32 {
// CHECK-NEXT:           %[[RES:.*]] = SpecHLS.lookUpTable [%arg0 ] :i32= {3334,4564,4564,3334,4564,3334,4564,3334 }
// CHECK:              return %[[RES:.*]] : i32
// CHECK:             }
// CHECK:             }
module {
func.func @bar(%a: i3) -> i32 {
        %res1 = SpecHLS.lookUpTable [%a]:i1 = {0,1,1,0,1,0,1,0}
        %res2 = SpecHLS.lookUpTable [%res1]:i32 = {3334,4564}
        return %res2 : i32
}
}
