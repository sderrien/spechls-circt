// RUN: spechls-opt --canonicalize %s | spechls-opt | FileCheck %s
// CHECK:module {
// CHECK:  hw.module @test1(out out0 : i32) {
// CHECK:    %c2222_i32 = hw.constant 2222 : i32
// CHECK:    hw.output %c2222_i32 : i32
// CHECK:  }
// CHECK:  hw.module @test2(out out0 : i32) {
// CHECK:    %c4444_i32 = hw.constant 4444 : i32
// CHECK:    hw.output %c4444_i32 : i32
// CHECK:  }
// CHECK:}

module {
  hw.module @test1(out out0 : i32) {
    %0 = hw.constant 1 : i2
    %1 = SpecHLS.lookUpTable [%0 :i2] :i32= {1111,2222,3333,4444 }
    hw.output %1 : i32
  }
  hw.module @test2(out out0 : i32) {
    %0 = hw.constant 3 : i2
    %1 = SpecHLS.lookUpTable [%0: i2] :i32= {1111,2222,3333,4444 }
    hw.output %1 : i32
  }

}
