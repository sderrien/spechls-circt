// RUN: spechls-opt --lower-spechls-to-comb %s | spechls-opt | FileCheck %

module
{

    hw.module @SCC0(out out_0 : i1)
    {
    // CHECK-LABEL: @SCC0
    // CHECK-NOT: SpecHLS.rollback
    // CHECK: SpecHLS.delay
    // CHECK: comb.mux
        %0 = SpecHLS.init @x0    : i1
        %1 = SpecHLS.init @stall : i1
        %2 = SpecHLS.init @idx   : i1
        %3 = SpecHLS.rollback %0:i1 by %2:i1 in [3 :i32] ctrl by %1
        hw.output %3 : i1
    }

    hw.module @SCC1(out out_0 : i1)
    {
    // CHECK-LABEL: @SCC0
    // CHECK-NOT: SpecHLS.rollback
    // CHECK: SpecHLS.delay
    // CHECK: comb.mux
        %0 = SpecHLS.init @x0    : i1
        %1 = SpecHLS.init @stall : i1
        %2 = SpecHLS.init @idx   : i2
        %3 = SpecHLS.rollback %0:i1 by %2:i2 in [3 :i32, 5 :i32] ctrl by %1
        hw.output %3 : i1
    }
}
