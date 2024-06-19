// RUN: spechls-opt --group-control --canonicalize %s | spechls-opt --yosys-optimizer="replace-with-optimized-module=true" | FileCheck %s

module {

  hw.module @SCC_0(out out_0 : i1) {
     // CHECK-LABEL: @SCC_0
    // CHECK: [[XOR1:%[0-9]+]] = SpecHLS.init @a : i1 
    %0 = SpecHLS.init @a : i1
    %1 = SpecHLS.init @b : i1
    // CHECK: [[XOR0:%[0-9]+]] = SpecHLS.init @c : i1 
    %2 = SpecHLS.init @c : i1
    %3 = SpecHLS.init @d : i1

    %6 = comb.and %0, %1 : i1
    %7 = comb.or %2, %1 : i1
    // CHECK-NOT: {{.*}} comb.xor {{.*}}
    %8 = comb.xor %2, %0 : i1

    // CHECK: [[CTRL:%.*]] = hw.instance "{{.*}}" @SCC_0ctrl_1_opt(in_0: [[XOR0]]: i1, in_1: [[XOR1]]: i1) -> (out_0: i1)
    // CHECK: {{.*}}SpecHLS.gamma @dummy [[CTRL]]:i1 ? {{.*}}
    %11 = SpecHLS.gamma @dummy %8 ? %3,%2 :i1
    %51 = SpecHLS.exit %11
    hw.output %51 :i1
  }


  hw.module @SCC_1(in %0 : i1, in %1 : i1, in %2 : i1, in %3 : i1, in %4 : i1, in %5 : i1, out out_0 : i1) {
    // CHECK-LABEL: @SCC_1
    
    //CHECK-NOT {{.*}}comb.and{{.*}}
    %6 = comb.and %0, %1 : i1
    
    //CHECK-NOT {{.*}}comb.or{{.*}}
    %7 = comb.or %2, %3 : i1
    
    //CHECK-NOT {{.*}}comb.xor{{.*}}
    %8 = comb.xor %6, %7 : i1

    //CHECK: [[CTRL:%.*]] = hw.instance {{.*}} @SCC_1ctrl_2_opt(in_0: %2: i1, in_1: %3: i1, in_2: %0: i1, in_3: %1: i1) -> ({{.*}}) 
    //CHECK: {{.*}}SpecHLS.gamma @x [[CTRL]]{{.*}} 
    %11 = SpecHLS.gamma @x %8 ? %4,%5 :i1
    hw.output %11 :i1
  }

  // CHECK-LABEL: @SCC_0ctrl_1_opt
  // CHECK-SAME: (in %in_0 : i1, in %in_1 : i1, out out_0 : i1)
  // CHECK-NEXT: %0 =  comb.xor %in_0, %in_1 : i1
  // CHECK-NEXT: hw.output %0{{.*}}

  // CHECK-LABEL: @SCC_1ctrl_2_opt
  // CHECK-SAME: (in [[ARG0:%.*]] : i1, in [[ARG1:%.*]] : i1, in [[ARG2:%.*]] : i1, in [[ARG3:%.*]] : i1, out [[OUT0:.*]] : i1)
  // CHECK: %0 = comb.and [[ARG2]], [[ARG3]] : i1
  // CHECK: %1 = comb.or [[ARG0]], [[ARG1]] : i1
  // CHECK: %2 = comb.xor %0, %1 : i1
  // CHECK: hw.output %2 : i1
}

