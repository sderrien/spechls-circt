// RUN: spechls-opt --merge-luts %s | spechls-opt | FileCheck %s
// CHECK:module {
// CHECK:  hw.module @bar(in %a : i3, out out0 : i32) {
// CHECK:    %LUT = SpecHLS.lookUpTable [%a : i3] :i32= {1234,3334,4564,3334,7896,3334,7896,1234 }
// CHECK:    hw.output %LUT : i32
// CHECK:  }
// CHECK:}

// module {
// hw.module @bar(in %a : i3, out out0 :i32) {
//         %res1 = SpecHLS.lookUpTable [%a : i3]:i2 = {0,1,2,1,3,1,3,0}
//         %res2 = SpecHLS.lookUpTable [%res1 : i2]:i32 = {1234,3334,4564,7896}
//         hw.output %res2 : i32
// }
// }

module {

//    hw.module @SCC_0_ctrl_0(in %in_0 : i2, in %in_1 : i1, out out_0 : i2) {
//     %LUT = SpecHLS.lookUpTable [%in_0 : i2] :i2= {0,0,1,2 }
//     %12 = comb.concat %LUT, %in_1 : i2, i1
//     %LUT_0 = SpecHLS.lookUpTable [%12 : i3] :i3= {0,0,1,1,2,3,4,4 }
//     %LUT_1 = SpecHLS.lookUpTable [%LUT_0 : i3] :i2= {0,0,0,1,2,2,2,2 }
//     hw.output %LUT_1 : i2
//   }

//  hw.module @SCC_0_ctrl_1(in %in_0 : i4, in %in_1 : i1, in %in_2 : i1, in %in_3 : i1, out out_0 : i1)  {
//    %LUT = SpecHLS.lookUpTable [%in_0 : i4] :i3= {0,0,1,2,3,3,4,4,5,5,6,6,6,6,6,6 }
//    %27 = comb.concat %LUT, %in_1 : i3, i1
//    %LUT_0 = SpecHLS.lookUpTable [%27 : i4] :i3= {0,0,1,1,2,2,3,4,5,5,6,6,7,7,7,7 }
//    %28 = comb.concat %LUT_0, %in_2 : i3, i1
//    %LUT_1 = SpecHLS.lookUpTable [%28 : i4] :i4= {0,0,1,1,2,2,3,3,4,5,6,6,7,7,8,8 }
//    %29 = comb.concat %LUT_1, %in_3 : i4, i1
//    %LUT_2 = SpecHLS.lookUpTable [%29 : i5] :i4= {0,0,1,1,2,2,3,3,4,4,5,5,6,7,8,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9 }
//    %LUT_3 = SpecHLS.lookUpTable [%LUT_2 : i4] :i3= {0,0,1,0,0,2,0,3,4,5,5,5,5,5,5,5 }
//    %LUT_4 = SpecHLS.lookUpTable [%LUT_3 : i3] :i1= {0,1,1,1,1,1,1,1 }
//    hw.output %LUT_4 : i1
//  }


//  hw.module @SCC_0_ctrl_1(in %in_0 : i4, in %in_1 : i1, in %in_2 : i1, in %in_3 : i1, out out_0 : i1)  {
//    %LUT = SpecHLS.lookUpTable [%in_0 : i4] :i3= {0,0,1,2,3,3,4,4,5,5,6,6,6,6,6,6 }
//    %27 = comb.concat %LUT, %in_1 : i3, i1
//    %LUT_0 = SpecHLS.lookUpTable [%27 : i4] :i3= {0,0,1,1,2,2,3,4,5,5,6,6,7,7,7,7 }
//    %28 = comb.concat %LUT_0, %in_2 : i3, i1
//    %LUT_1 = SpecHLS.lookUpTable [%28 : i4] :i4= {0,0,1,1,2,2,3,3,4,5,6,6,7,7,8,8 }
//    %29 = comb.concat %LUT_1, %in_3 : i4, i1
//    %LUT_2 = SpecHLS.lookUpTable [%29 : i5] :i4= {0,0,1,1,2,2,3,3,4,4,5,5,6,7,8,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9 }
//    %LUT_3 = SpecHLS.lookUpTable [%LUT_2 : i4] :i3= {0,0,1,0,0,2,0,3,4,5,5,5,5,5,5,5 }
//    %LUT_4 = SpecHLS.lookUpTable [%LUT_3 : i3] :i1= {0,1,1,1,1,1,1,1 }
//    hw.output %LUT_4 : i1
//  }

    hw.module @test5_golden(in %in : i4, out out_0 : i2)  {

      %in_0 = comb.extract %in from 0 : (i4) -> i2
      %in_1 = comb.extract %in from 2 : (i4) -> i1
      %in_2 = comb.extract %in from 3 : (i4) -> i1
      %LUT_1 = SpecHLS.lookUpTable [%in_0 : i2] :i1= {0,1,0,1}
      %29 = comb.concat %LUT_1, %in_1,%in_2 : i1, i1, i1
      %LUT_2 = SpecHLS.lookUpTable [%29 : i3] :i2= {1,3,1,1,0,2,3,3}
      hw.output %LUT_2 : i2
    }

    hw.module @test5_opt(in %in : i4, out out_0 : i2)  {
      %in_0 = comb.extract %in from 0 : (i4) -> i2
      %in_1 = comb.extract %in from 2 : (i4) -> i1
      %in_2 = comb.extract %in from 3 : (i4) -> i1
      %0 = comb.concat %in_0, %in_1, %in_2 : i2, i1, i1
      %LUT = SpecHLS.lookUpTable [%0 : i4] :i2= {0,3,3,0,1,1,1,1,0,2,2,0,3,3,3,3 }
      hw.output %LUT : i2
    }
//}

//hw.module @SCC_0_ctrl_1()  {
//
//   %0 = hw.constant 0 : i4
//   %1 = hw.constant 1 : i4
//   %2 = hw.constant 2 : i4
//   %3 = hw.constant 3 : i4
//   %4 = hw.constant 4 : i4
//   %5 = hw.constant 5 : i4
//   %6 = hw.constant 6 : i4
//   %7 = hw.constant 7 : i4
//   %8 = hw.constant 8 : i4
//   %9 = hw.constant 9 : i4
//   %10 = hw.constant 10 : i4
//   %11 = hw.constant 11 : i4
//   %12 = hw.constant 12 : i4
//   %13 = hw.constant 13 : i4
//   %14 = hw.constant 14 : i4
//   %15 = hw.constant 15 : i4
//
//
//  %res_0 = hw.instance "%1" @test5_golden(in_0: %0: i4) -> (out0: i1)
//  %res_0 = hw.instance "%1" @test5_golden(in_0: %0: i4) -> (out0: i1)
//  %res_0 = hw.instance "%1" @test5_golden(in_0: %0: i4) -> (out0: i1)
//  %res_0 = hw.instance "%1" @test5_golden(in_0: %0: i4) -> (out0: i1)
//  %res_0 = hw.instance "%1" @test5_golden(in_0: %0: i4) -> (out0: i1)
//  %res_0 = hw.instance "%1" @test5_golden(in_0: %0: i4) -> (out0: i1)
//  %res_0 = hw.instance "%1" @test5_golden(in_0: %0: i4) -> (out0: i1)
//  %res_0 = hw.instance "%1" @test5_golden(in_0: %0: i4) -> (out0: i1)
//  %res_0 = hw.instance "%1" @test5_golden(in_0: %0: i4) -> (out0: i1)
//
//}

}
