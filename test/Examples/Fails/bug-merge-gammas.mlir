module {

//  hw.module @gamma3(in %in_0 : ui3, in %in_1 : i32, in %in_2 : i32, in %in_3 : i32, in %in_4 : i32, in %in_5 : ui2, in %in_6 : i32, in %in_7 : i32, in %in_8 : i32, in %in_9 : i1, in %in_10 : i32, out out_0 : i32) attributes {"#pragma" = "GAMMA_GROUP"} {
//    %gamma = SpecHLS.gamma @x %in_5:ui2 ? %in_6,%in_7,%in_8 :i32
//    %gamma_0 = SpecHLS.gamma @x %in_0:ui3 ? %in_1,%in_2,%gamma,%in_3,%in_4 :i32
//    %gamma_1 = SpecHLS.gamma @x %in_9:i1 ? %in_10,%gamma_0 :i32
//    hw.output %gamma_1 : i32
//  }

  hw.module @gamma4(in %in_0 : ui3, in %in_1 : i32, in %in_2 : i32, in %in_3 : i32, in %in_4 : i32, in %in_5 : ui2, in %in_6 : i32, in %in_7 : i32, in %in_8 : i32, in %in_9 : i1, in %in_10 : i32, out out_0 : i32) attributes {"#pragma" = "GAMMA_GROUP"} {
    %gamma = SpecHLS.gamma @x %in_5:ui2 ? %in_6,%in_7,%in_8 :i32
    %gamma_0 = SpecHLS.gamma @x %in_0:ui3 ? %in_1,%in_2,%gamma,%in_3,%in_4 :i32
    hw.output %gamma_0 : i32
  }

//  hw.module @gamma_0(in %in_0 : ui3, in %in_1 : i32, in %in_2 : i32, in %in_3 : i32, in %in_4 : i32, in %in_5 : i1, in %in_6 : i32, in %in_7 : i32, in %in_8 : i1, in %in_9 : i32, out out_0 : i32) attributes {"#pragma" = "GAMMA_GROUP"} {
//    %gamma = SpecHLS.gamma @x %in_5:i1 ? %in_6,%in_7 :i32
//    %gamma_0 = SpecHLS.gamma @x %in_0:ui3 ? %in_1,%in_2,%in_3,%in_4,%gamma :i32
//    %gamma_1 = SpecHLS.gamma @x %in_8:i1 ? %in_9,%gamma_0 :i32
//    hw.output %gamma_1 : i32
//  }
//
//  hw.module @gamma_1(in %in_0 : i1, in %in_1 : i32, in %in_2 : i1, in %in_3 : i32, in %in_4 : i32, in %in_5 : i1, in %in_6 : i32, out out_0 : i32) attributes {"#pragma" = "GAMMA_GROUP"} {
//    %gamma = SpecHLS.gamma @y %in_2:i1 ? %in_3,%in_4 :i32
//    %gamma_0 = SpecHLS.gamma @y %in_0:i1 ? %in_1,%gamma :i32
//    %gamma_1 = SpecHLS.gamma @y %in_5:i1 ? %in_6,%gamma_0 :i32
//    hw.output %gamma_1 : i32
//  }
//*/
}

