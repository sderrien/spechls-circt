// RUN: spechls-opt --merge-gammas %s | spechls-opt | FileCheck %s
// CHECK-LABEL:   @bar
// CHECK-SAME: (in %arg0: i2, in %arg1: i2, in %arg2: i32, in %arg3: i32,in  %arg4: i32, in %arg5: i32, out out0 :i32) {
// CHECK-NEXT:     %0 = comb.concat %arg1, %arg0 : i2, i2
// CHECK:     %1 = SpecHLS.lookUpTable [%0 ] :i3= {0,1,2,3,4,4,4,4,5,5,5,5,6,6,6,6 }
// CHECK:     %2 = SpecHLS.gamma %1 ? %arg2,%arg3,%arg4,%arg2,%arg4 :i32
// CHECK:    hw.output %2 : i32

module {

 
 
 
 
 

    hw.module @test2(in %s0 : i1,in %s1 : i1,in %a : i32, in %b: i32,in %c: i32,in %d:i32, out out0 :i32) {

        %10 = SpecHLS.gamma @x %s0 ? %a, %b :i32
        %11 = SpecHLS.gamma @X %s0 ? %c, %d :i32
        %12 = SpecHLS.gamma @x %s1 ? %10, %11 :i32

        hw.output %12 : i32
    }


   hw.module @test3(in %s0 : i1,in %s1 : i1, in %s2 : i1, in %s3 : i1, in %a : i32, in %b: i32,in %c: i32,in %d:i32,in %e: i32,in %f:i32,in %g:i32, out out0 :i32) {

       %10 = SpecHLS.gamma @x %s0 ? %a, %b :i32
       %11 = SpecHLS.gamma @X %s1 ? %c, %10 :i32
       %12 = SpecHLS.gamma @X %s2 ? %d, %e :i32
       %13 = SpecHLS.gamma @X %s3 ? %11, %12 :i32
       %14 = SpecHLS.gamma @x %s2 ? %f, %13 :i32

       hw.output %14 : i32
   }

   hw.module @test4(in %s0 : i1,in %s1 : i1, in %s2 : i1, in %a : i32, in %b: i32,in %c: i32,in %d:i32, out out0 :i32) {

       %10 = SpecHLS.gamma @x %s0 ? %a, %b :i32
       %11 = SpecHLS.gamma @X %s0 ? %c, %d :i32
       %12 = SpecHLS.gamma @x %s1 ? %10, %11 :i32

       hw.output %12 : i32
   }

}

