// RUN: spechls-opt --merge-gammas %s | spechls-opt | FileCheck %s
module {
  hw.module @SCC_0(in %sel : i2,in %1 : i32,in %2 : i32,in %3 : i32,out out0 :i32) {

    %48 = SpecHLS.gamma @x %sel:i2 ? %1,%2,%3,%3 :i32

    hw.output %48 : i32
  }
}
