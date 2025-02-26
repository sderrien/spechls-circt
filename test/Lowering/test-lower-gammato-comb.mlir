// RUN: spechls-opt --lower-spechls-to-comb %s | spechls-opt | FileCheck %

module
{

    hw.module @SCC0(
    in %sel : i4,
    in %in_0 : i32,
    in %in_1 : i32,
    in %in_2 : i32,
    in %in_3 : i32,
    in %in_4 : i32,
    in %in_5 : i32,
    in %in_6 : i32,
    in %in_7 : i32,
    in %in_8 : i32,
    out out_0 : i32)
    {
        %gamma = SpecHLS.gamma @LUT %sel:i4 ? %in_0,%in_1,%in_2,%in_3,%in_4,%in_5,%in_6,%in_7,%in_8:i32
        hw.output %gamma : i32
    }

}
