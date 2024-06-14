// RUN: spechls-opt --merge-luts %s | spechls-opt | FileCheck %s
// CHECK:module {
// CHECK:  hw.module @bar(in %a : i3, out out0 : i32) {
// CHECK:    %LUT = SpecHLS.lookUpTable [%a : i3] :i32= {1234,3334,4564,3334,7896,3334,7896,1234 }
// CHECK:    hw.output %LUT : i32
// CHECK:  }
// CHECK:}

module {
hw.module @bar(in %a : i3, out out0 :i32) {
        %res1 = SpecHLS.lookUpTable [%a : i3]:i2 = {0,1,2,1,3,1,3,0}
        %res2 = SpecHLS.lookUpTable [%res1 : i2]:i32 = {1234,3334,4564,7896}
        hw.output %res2 : i32
}
}
