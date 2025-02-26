// RUN: spechls-opt --barrett-reduction %s | FileCheck %s

// This test verifies that the Barrett Reduction pass correctly transforms
// comb.mod operations with constant operands within hw.module into a
// Barrett reduction sequence.

module {
// Define a hardware module using CIRCT's hw.module with comb.mod operations
hw.module @test_barrett_10(in %in0: i32, out o: i32) {
  // Perform input mod 10 using comb.mod
  %mod_result = comb.mod %in0, 10 : i32
  // Output the mod_result
  hw.output %mod_result : i32

  // CHECK-LABEL: hw.module @test_barrett_10(
  // CHECK: comb.mul
  // CHECK: comb.shru
  // CHECK: comb.mul
  // CHECK: comb.sub
  // CHECK: comb.icmp
  // CHECK: comb.mux
  // CHECK-NOT: comb.mod
}

hw.module @test_barrett_13(in %in0: i32, out o: i32) {
  // Perform input mod 13 using comb.mod
  %mod_result = comb.mod %in0, 13 : i32
  // Output the mod_result
  hw.output %mod_result : i32

  // CHECK-LABEL: hw.module @test_barrett_13(
  // CHECK: comb.mul
  // CHECK: comb.shru
  // CHECK: comb.mul
  // CHECK: comb.sub
  // CHECK: comb.icmp
  // CHECK: comb.mux
  // CHECK-NOT: comb.mod
}

hw.module @test_barrett_void(in %in0: i32, out o: i32) {
  // Perform input mod 7 using comb.mod
  %mod_result = comb.mod %in0, 7 : i32
  // Output the mod_result
  hw.output %mod_result : i32

  // CHECK-LABEL: hw.module @test_barrett_void(
  // CHECK: comb.mul
  // CHECK: comb.shru
  // CHECK: comb.mul
  // CHECK: comb.sub
  // CHECK: comb.icmp
  // CHECK: comb.mux
  // CHECK-NOT: comb.mod
}

hw.module @test_barrett_multiple_mods(in %in0: i32, in %in1: i32, out o0: i32, out o1: i32) {
  // Perform multiple mod operations
  %mod_result0 = comb.mod %in0, 5 : i32
  %mod_result1 = comb.mod %in1, 3 : i32
  // Output the mod_results
  hw.output %mod_result0 : i32
  hw.output %mod_result1 : i32

  // CHECK-LABEL: hw.module @test_barrett_multiple_mods(
  // CHECK: comb.mul
  // CHECK: comb.shru
  // CHECK: comb.mul
  // CHECK: comb.sub
  // CHECK: comb.icmp
  // CHECK: comb.mux
  // CHECK-NOT: comb.mod
  // CHECK: comb.mul
  // CHECK: comb.shru
  // CHECK: comb.mul
  // CHECK: comb.sub
  // CHECK: comb.icmp
  // CHECK: comb.mux
  // CHECK-NOT: comb.mod
}

hw.module @test_barrett_no_mod(in %in0: i32, out o: i32) {
  // No comb.mod operation here
  %const = arith.constant 42 : i32
  hw.output %const : i32

  // CHECK-LABEL: hw.module @test_barrett_no_mod(
  // CHECK: arith.constant 42 : i32
  // CHECK: hw.output %const : i32
  // CHECK-NOT: comb.mod
  // CHECK-NOT: comb.mul
  // CHECK-NOT: comb.shru
}

hw.module @test_barrett_complex(in %in0: i32, in %in1: i32, out o0: i32, out o1: i32) {
  %1 = arith.constant 2 : i32
  %we = arith.constant 1 : i1
  %mod0 = comb.mod %in0, 15 : i32
  %mod1 = comb.mod %in1, 7 : i32
  hw.output %mod0 : i32
  hw.output %mod1 : i32

  // CHECK-LABEL: hw.module @test_barrett_complex(
  // CHECK: comb.mul
  // CHECK: comb.shru
  // CHECK: comb.mul
  // CHECK: comb.sub
  // CHECK: comb.icmp
  // CHECK: comb.mux
  // CHECK-NOT: comb.mod
  // CHECK: comb.mul
  // CHECK: comb.shru
  // CHECK: comb.mul
  // CHECK: comb.sub
  // CHECK: comb.icmp
  // CHECK: comb.mux
  // CHECK-NOT: comb.mod
}

hw.module @test_barrett_mod_zero(in %in0: i32, out o: i32) {
  // Perform input mod 0 using comb.mod (should handle gracefully)
  %mod_result = comb.mod %in0, 0 : i32
  // Output the mod_result
  hw.output %mod_result : i32

  // CHECK-LABEL: hw.module @test_barrett_mod_zero(
  // CHECK: comb.mod
  // CHECK-NOT: comb.mul
  // CHECK-NOT: comb.shru
}

}

