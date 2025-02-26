// RUN: spechls-opt --canonicalize %s | spechls-opt | FileCheck %s
module {

  hw.module @SCC_0(
     in %enable : i1, in %value : i32, in %address : i32, in %array:memref<16xi32>,
   out result : memref<16xi32>)

   {
    %index = arith.index_cast %address : i32 to index
    %mu = SpecHLS.mu @x : %array, %array : memref<16xi32>
    %31 = SpecHLS.alpha @x : %enable -> %mu[%index], %value : memref<16xi32>
    hw.output %31 :memref<16xi32>
  }
}
