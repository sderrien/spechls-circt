// RUN: spechls-opt --merge-gammas --canonicalize %s | spechls-opt | FileCheck %s

module {

hw.module @SCC_0() {
// CHECK-LABEL: SCC_0
// CHECK-NOT: {{.*}}SpecHLS.gamma{{.*}}
// CHECK: %LUT{{(_[0-9]+)?}} = SpecHLS.lookUpTable {{.*}}= {0,1,2,2,3,3,4,4 }
// CHECK: %gamma = SpecHLS.gamma {{.*}}? %1,%0,%2,%3{{[ ]*}}:i1
// CHECK-NOT: {{.*}}SpecHLS.gamma{{.*}}
  %0 = SpecHLS.init @x0 : i1
  %1 = SpecHLS.init @x1 : i1
  %2 = SpecHLS.init @x2 : i1
  %3 = SpecHLS.init @c0 : i1
  %4 = SpecHLS.gamma @g0 %3 ? %1,%0 :i1

  %10 = SpecHLS.init @c1 : i2
  %11 = SpecHLS.gamma @g1 %10 ? %4,%2,%3 :i1
  %12 = SpecHLS.exit %11
  hw.output
}

hw.module @SCC_1() {
// CHECK-LABEL: SCC_1
// CHECK-NOT: {{.*}}SpecHLS.gamma{{.*}}
// CHECK: %LUT{{(_[0-9]+)?}} = SpecHLS.lookUpTable {{.*}}= {0,0,1,2,3,3,4,4 }
// CHECK: %gamma = SpecHLS.gamma {{.*}}? %2,%1,%0,%3{{[ ]*}}:i1
// CHECK-NOT: {{.*}}SpecHLS.gamma{{.*}}
  %0 = SpecHLS.init @x0 : i1
  %1 = SpecHLS.init @x1 : i1
  %2 = SpecHLS.init @x2 : i1
  %3 = SpecHLS.init @c0 : i1
  %4 = SpecHLS.gamma @g0 %3 ? %1,%0 :i1

  %10 = SpecHLS.init @c1 : i2
  %11 = SpecHLS.gamma @g1 %10 ? %2,%4,%3 :i1
  %12 = SpecHLS.exit %11
  hw.output
}

hw.module @SCC_2() {
// CHECK-LABEL: SCC_2
// CHECK-NOT: {{.*}}SpecHLS.gamma{{.*}}
// CHECK: %LUT{{(_[0-9]+)?}} = SpecHLS.lookUpTable {{.*}}= {0,0,1,1,2,3,4,4 }
// CHECK: %gamma = SpecHLS.gamma {{.*}}? %2,%3,%1,%0{{[ ]*}}:i1
// CHECK-NOT: {{.*}}SpecHLS.gamma{{.*}}
  %0 = SpecHLS.init @x0 : i1
  %1 = SpecHLS.init @x1 : i1
  %2 = SpecHLS.init @x2 : i1
  %3 = SpecHLS.init @c0 : i1
  %4 = SpecHLS.gamma @g0 %3 ? %1,%0 :i1

  %10 = SpecHLS.init @c1 : i2
  %11 = SpecHLS.gamma @g1 %10 ? %2,%3,%4 :i1
  %12 = SpecHLS.exit %11
  hw.output
}

hw.module @SCC_3 () {
// CHECK-LABEL: SCC_3
// CHECK-NOT: {{.*}}SpecHLS.gamma{{.*}}
// CHECK: %LUT = SpecHLS.lookUpTable {{.*}}= {0,1,2,2,3,3,4,4 }
// CHECK: %LUT{{(_[0-9]+)?}} = SpecHLS.lookUpTable {{.*}}= {0,0,1,1,2,2,3,4,5,5,6,6,7,7,8,8 }
// CHECK: %gamma = SpecHLS.gamma {{.*}}? %0,%1,%6,%3,%4{{[ ]*}}:i1
// CHECK-NOT: {{.*}}SpecHLS.gamma{{.*}}
  %0 = SpecHLS.init @x0 : i1
  %1 = SpecHLS.init @x1 : i1
  %2 = SpecHLS.init @c0 : i1
  
  %3 = SpecHLS.init @x2 : i1
  %4 = SpecHLS.init @x3 : i1
  %5 = SpecHLS.init @c1 : i1
  
  %6 = SpecHLS.init @x4 : i1
  %7 = SpecHLS.init @c2 : i2

  %8 = SpecHLS.gamma @g0 %2 ? %0,%1 :i1
  %9 = SpecHLS.gamma @g1 %5 ? %3,%4 :i1
  %10 = SpecHLS.gamma @g2 %7 ? %8,%6,%9 :i1
  %11 = SpecHLS.exit %10
  hw.output
}

}

