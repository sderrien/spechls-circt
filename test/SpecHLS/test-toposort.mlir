// RUN: spechls-opt --merge-gammas %s | spechls-opt | FileCheck %s
module {
  hw.module @SCC_0() {
    %false = hw.constant false
    %true = hw.constant true
    %0 = SpecHLS.init @io_state : i32
    %2 = SpecHLS.init @guard : i1
    %4 = SpecHLS.init @done : i1
    %5 = SpecHLS.mu @done : %4, %2 : i1
    hw.output
  }
}
