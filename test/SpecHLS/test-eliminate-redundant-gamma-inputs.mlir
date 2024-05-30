// RUN: spechls-opt --eliminate-redundant-gamme-inputs %s | spechls-opt | FileCheck %s
// CHECK-LABEL: test1
// CHECK: %0 = SpecHLS.gamma @x %[[ARG0:.*]] ? [[ARG1:.*]], [[ARG2:.*]] -> i32
// CHECK: return %0 : i32

module
{
  func.func @test1(%arg0: i2, %arg1 : i32, %arg2 : i32) -> i32
  {
    %0 = SpecHLS.gamma @gamma0 %arg0 ? %arg1, %arg2, %arg1, %arg2 : i32
    return %0 : i32
  }

  func.func @test2(%arg0: i2, %arg1 : i32) -> i32
  {
    %0 = SpecHLS.gamma @gamma0 %arg0 ? %arg1, %arg1, %arg1 : i32
    return %0 : i32
  }

  func.func @test3(%arg0 : i1, %arg1 : i32, %arg2 : i32) -> i32
  {
    %0 = SpecHLS.gamma @gamma0 %arg0 ? %arg1, %arg2 : i32
    %1 = SpecHLS.gamma @gamma1 %arg0 ? %arg2, %0 : i32
    return %1 : i32
  }
}
