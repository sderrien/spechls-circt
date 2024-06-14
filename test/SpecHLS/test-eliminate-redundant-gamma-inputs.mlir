// RUN: spechls-opt --merge-gammas --eliminate-redundant-gamma-inputs %s | spechls-opt | FileCheck %s

module
{
  func.func @test1(%arg0: i2, %arg1 : i32, %arg2 : i32) -> i32
  {
// CHECK-LABEL: test1
// CHECK-SAME: {{.*}}%{{.*}}[[ARG1:%[a-z0-9]*]]{{.*}}[[ARG2:%[a-z0-9]*]]{{.*}}
// CHECK: [[RES:%gamma]] = SpecHLS.gamma @gamma0 {{.*}} ? [[ARG1]],[[ARG2]] :i32
// CHECK: return [[RES]] : i32
    %0 = SpecHLS.gamma @gamma0 %arg0 ? %arg1, %arg2, %arg1, %arg2 : i32
    return %0 : i32
  }

  func.func @test2(%arg0: i2, %arg1 : i32) -> i32
  {
// CHECK-LABEL: test2
// CHECK-SAME: {{.*}}%{{.*}}[[ARG1:%[a-z]*[0-9]+]]{{.*}}
// CHECK: return [[ARG1]]{{.*}}
    %0 = SpecHLS.gamma @gamma0 %arg0 ? %arg1, %arg1, %arg1 : i32
    return %0 : i32
  }

  func.func @test3(%arg0 : i1, %arg1 : i32, %arg2 : i32) -> i32
  {
// CHECK-LABEL: test3
// CHECK-SAME: {{.*}}%{{.*}}[[ARG1:%[a-z]*[0-9]+]]{{.*}}[[ARG2:%[a-z]*[0-9]+]]{{.*}}
// CHECK: [[RES:%gamma]] = SpecHLS.gamma{{.*}}? [[ARG2]],[[ARG1]] :i32
// CHECK-NOT: {{.*}} SpecHLS.gamma {{.*}}
// CHECK: return [[RES]]{{.*}}
    %0 = SpecHLS.gamma @gamma0 %arg0 ? %arg1, %arg2 : i32
    %1 = SpecHLS.gamma @gamma1 %arg0 ? %arg2, %0 : i32
    return %1 : i32
  }
}
